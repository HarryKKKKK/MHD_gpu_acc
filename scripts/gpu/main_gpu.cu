#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/grid_gpu.cuh"
#include "gpu/solver_gpu.cuh"
#include "init.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

// ============================================================
// CSV output — all 9 MHD fields + derived pressure
// ============================================================

template<typename FieldFn>
static void write_field_csv(
    const std::vector<Conserved>& data,
    int total_nx, int nx, int ny, int ng,
    const std::string& path,
    FieldFn field_fn
) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open output file: " + path);
    // max_digits10 (17): enough decimal digits to round-trip a double exactly,
    // so CPU/GPU/MPI CSV diffs reflect the true underlying bit pattern instead
    // of being masked by output rounding.
    f << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int j = ng + ny - 1; j >= ng; --j) {
        for (int i = ng; i < ng + nx; ++i) {
            if (i > ng) f << ',';
            f << field_fn(data[static_cast<std::size_t>(j * total_nx + i)]);
        }
        f << '\n';
    }
}

static void write_all_fields(
    const Grid2DGPU& gpu_grid,
    const std::string& dir,
    const std::string& prefix
) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);

    std::vector<Conserved> host_data;
    gpu_grid.download_to_aos(host_data);

    const int total_nx = gpu_grid.total_nx();
    const int nx       = gpu_grid.nx();
    const int ny       = gpu_grid.ny();
    const int ng       = gpu_grid.ng();

    const auto path = [&](const char* name) {
        return dir + "/" + prefix + "_" + name + ".csv";
    };

    // Primitive velocity and pressure (derived)
    write_field_csv(host_data, total_nx, nx, ny, ng, path("rho"),
        [](const Conserved& U){ return U.rho; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("u"),
        [](const Conserved& U){ return U.rhou / U.rho; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("v"),
        [](const Conserved& U){ return U.rhov / U.rho; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("w"),
        [](const Conserved& U){ return U.rhow / U.rho; });
    // Magnetic fields
    write_field_csv(host_data, total_nx, nx, ny, ng, path("Bx"),
        [](const Conserved& U){ return U.Bx; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("By"),
        [](const Conserved& U){ return U.By; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("Bz"),
        [](const Conserved& U){ return U.Bz; });
    // Total energy and GLM scalar
    write_field_csv(host_data, total_nx, nx, ny, ng, path("E"),
        [](const Conserved& U){ return U.E; });
    write_field_csv(host_data, total_nx, nx, ny, ng, path("psi"),
        [](const Conserved& U){ return U.psi; });
    // Pressure (derived via cons_to_prim — host-side, uses phys::gamma inline var)
    write_field_csv(host_data, total_nx, nx, ny, ng, path("p"),
        [](const Conserved& U){ return phys::cons_to_prim(U).p; });

    std::cout << "  Wrote fields to " << dir << "/" << prefix << "_*.csv\n";
}

static std::uint64_t interior_state_hash(const Grid2DGPU& gpu_grid) {
    std::vector<Conserved> host_data;
    gpu_grid.download_to_aos(host_data);

    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix_double = [&](double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (bits >> (8 * byte)) & 0xffULL;
            hash *= 1099511628211ULL;
        }
    };

    const int total_nx = gpu_grid.total_nx();
    for (int j = gpu_grid.j_begin(); j < gpu_grid.j_end(); ++j) {
        for (int i = gpu_grid.i_begin(); i < gpu_grid.i_end(); ++i) {
            const Conserved& U =
                host_data[static_cast<std::size_t>(j * total_nx + i)];
            mix_double(U.rho);
            mix_double(U.rhou);
            mix_double(U.rhov);
            mix_double(U.rhow);
            mix_double(U.Bx);
            mix_double(U.By);
            mix_double(U.Bz);
            mix_double(U.E);
            mix_double(U.psi);
        }
    }
    return hash;
}

// ============================================================
// Argument parsing
// ============================================================

struct RunConfig {
    std::string   case_name     = "kelvin_helmholtz";
    int           n_scale       = 1;
    RiemannSolver solver        = RiemannSolver::HLLD;
    std::string   out_dir       = "output";
    bool          write_out     = false;
    int           warmup_steps  = 0;
    int           benchmark_steps = 0;
};

static RunConfig parse_args(int argc, char** argv) {
    RunConfig rc;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            rc.case_name = argv[++i];
        } else if (arg == "--solver" && i + 1 < argc) {
            std::string s = argv[++i];
            if      (s == "hll")   rc.solver = RiemannSolver::HLL;
            else if (s == "hllc")  rc.solver = RiemannSolver::HLLC;
            else if (s == "hlld")  rc.solver = RiemannSolver::HLLD;
            else if (s == "force") rc.solver = RiemannSolver::FORCE;
            else throw std::runtime_error("Unknown solver: " + s);
        } else if (arg == "--out" && i + 1 < argc) {
            rc.out_dir   = argv[++i];
            rc.write_out = true;
        } else if (arg == "--output") {
            rc.write_out = true;
        } else if (arg == "--no-out") {
            rc.write_out = false;
        } else if (arg == "--warmup-steps" && i + 1 < argc) {
            rc.warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--benchmark-steps" && i + 1 < argc) {
            rc.benchmark_steps = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            rc.n_scale = std::stoi(arg);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
    if (rc.warmup_steps < 0)
        throw std::runtime_error("--warmup-steps must be non-negative");
    if (rc.benchmark_steps < 0)
        throw std::runtime_error("--benchmark-steps must be non-negative");
    if (rc.benchmark_steps == 0 && rc.warmup_steps != 0)
        throw std::runtime_error(
            "--warmup-steps requires a positive --benchmark-steps");
    return rc;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    RunConfig rc;
    try {
        rc = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    CaseConfig cfg;
    try {
        cfg = get_n_case_config(rc.case_name, rc.n_scale);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Set the host-side gamma for cons_to_prim calls in write_all_fields.
    // The device-side gamma is set via set_gpu_physics_gamma below.
    phys::gamma = cfg.gamma;

    const GpuLaunchConfig launch_cfg = get_gpu_launch_config();
    const bool benchmark_mode = rc.benchmark_steps > 0;

    std::cout << "=== MHD GLM GPU Solver ===\n";
    std::cout << "  Case   : " << rc.case_name << "\n";
    std::cout << "  n      : " << rc.n_scale << "\n";
    std::cout << "  Gamma  : " << cfg.gamma << "\n";
    std::cout << "  Solver : "
              << (rc.solver == RiemannSolver::HLL  ? "HLL"  :
                  rc.solver == RiemannSolver::HLLC ? "HLLC" :
                  rc.solver == RiemannSolver::HLLD ? "HLLD" : "FORCE") << "\n";
    std::cout << "[GPU] nx: "          << cfg.nx            << "\n";
    std::cout << "[GPU] ny: "          << cfg.ny            << "\n";
    std::cout << "[GPU] total_cells: " << (cfg.nx * cfg.ny) << "\n";
    std::cout << "[GPU] advance_x launch: "
              << launch_cfg.x_block_x << "x" << launch_cfg.x_block_y
              << ", min_blocks_per_sm="
              << launch_cfg.x_min_blocks_per_sm << "\n";
    std::cout << "[GPU] advance_y launch: "
              << launch_cfg.y_block_x << "x" << launch_cfg.y_block_y
              << ", min_blocks_per_sm="
              << launch_cfg.y_min_blocks_per_sm << "\n";
    if (benchmark_mode) {
        std::cout << "[GPU] tuning mode: warmup_steps=" << rc.warmup_steps
                  << ", benchmark_steps=" << rc.benchmark_steps << "\n";
    }
    std::cout << std::flush;  // flush now: Slurm stdout is fully buffered (not a tty)

    // Build initial CPU grid then upload to GPU.
    Grid2D cpu_grid = make_n_grid(rc.case_name, rc.n_scale);

    Grid2DGPU Uold(cfg.nx, cfg.ny, cfg.ng,
                   cfg.x_min, cfg.x_max,
                   cfg.y_min, cfg.y_max);
    Uold.upload_from_aos(cpu_grid.data());

    Grid2DGPU Unew(cfg.nx, cfg.ny, cfg.ng,
                   cfg.x_min, cfg.x_max,
                   cfg.y_min, cfg.y_max);
    Grid2DGPU Utmp(cfg.nx, cfg.ny, cfg.ng,
                   cfg.x_min, cfg.x_max,
                   cfg.y_min, cfg.y_max);

    GpuWorkspace ws;
    init_gpu_workspace(ws, Uold);

    // Propagate physics constants to device memory.
    set_gpu_physics_gamma(cfg.gamma);
    set_gpu_physics_ch(0.0);  // ch_glm starts at 0; updated by compute_dt_gpu

    if (rc.write_out)
        write_all_fields(Uold, rc.out_dir, rc.case_name + "_gpu_t0");

    const bool has_snaps = !cfg.snapshot_times.empty();
    std::size_t snap_idx = 0;

    double t    = 0.0;
    int    step = 0;

    GpuAdvanceTimings advance_timings;
    bool benchmark_wall_started = false;
    std::chrono::steady_clock::time_point benchmark_wall_start;

    auto wall_start = std::chrono::steady_clock::now();

    while (benchmark_mode || t < cfg.t_end) {
        if (benchmark_mode &&
            step >= rc.warmup_steps + rc.benchmark_steps) {
            break;
        }

        const bool measured_step =
            benchmark_mode && step >= rc.warmup_steps;
        if (measured_step && !benchmark_wall_started) {
            benchmark_wall_start = std::chrono::steady_clock::now();
            benchmark_wall_started = true;
        }

        // Determine the next time we must not overshoot:
        // either the next snapshot or t_end, whichever is sooner.
        double t_next = cfg.t_end;
        if (!benchmark_mode && has_snaps && snap_idx < cfg.snapshot_times.size())
            t_next = std::min(t_next, cfg.snapshot_times[snap_idx]);

        const double dt_raw = compute_dt_gpu(Uold, ws, cfg.cfl);
        // Tuning mode deliberately executes an exact step count, independent
        // of the case's reporting horizon.  Normal simulations still clamp
        // the final step to the next snapshot or t_end.
        const double dt = benchmark_mode
            ? dt_raw
            : std::min(dt_raw, t_next - t);

        // Guard against numerical blowup: dt should never be zero or non-finite.
        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "[ERROR] dt=" << dt << " at step=" << step
                      << " t=" << t << " (dt_raw=" << dt_raw << "). Aborting.\n";
            break;
        }
        // Warn if dt is suspiciously tiny relative to t_end (signal speed blowup).
        // A dt more than 8 orders of magnitude below t_end/N_expected is pathological.
        const double dt_floor = 1.0e-8 * cfg.t_end;
        if (dt < dt_floor) {
            std::cerr << "[WARN]  dt=" << std::scientific << dt
                      << " (floor=" << dt_floor << ") at step=" << step
                      << " t=" << t << " — numerical blowup, stopping.\n";
            break;
        }

        advance_gpu(
            Uold, Utmp, Unew, ws, dt, rc.solver, cfg.bc,
            measured_step ? &advance_timings : nullptr);

        Uold.swap(Unew);
        t    += dt;
        step += 1;

        // Periodic progress output so the Slurm log is never silent for long.
        if (!benchmark_mode && step % 500 == 0) {
            const double elapsed_so_far =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wall_start).count();
            std::cout << "  [step " << std::setw(6) << step
                      << "]  t=" << std::scientific << std::setprecision(4) << t
                      << "  dt=" << dt
                      << "  wall=" << std::fixed << std::setprecision(1)
                      << elapsed_so_far << "s\n" << std::flush;
        }

        // Write any snapshots whose time we have just reached.
        if (!benchmark_mode && has_snaps) {
            while (snap_idx < cfg.snapshot_times.size() &&
                   t >= cfg.snapshot_times[snap_idx] - 1e-12) {
                std::cout << "  [snap] " << cfg.snapshot_tags[snap_idx]
                          << "  t_phys=" << std::scientific << std::setprecision(6)
                          << t << " s\n";
                if (rc.write_out) {
                    write_all_fields(Uold, rc.out_dir,
                        rc.case_name + "_gpu_" + cfg.snapshot_tags[snap_idx]);
                }
                ++snap_idx;
            }
        }
    }

    if (benchmark_mode) {
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            std::cerr << "[ERROR] CUDA synchronization failed after benchmark: "
                      << cudaGetErrorString(sync_status) << "\n";
            free_gpu_workspace(ws);
            return 1;
        }
    }

    auto wall_end = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << "[GPU] Total steps= " << step << "\n";
    std::cout << "  Elapsed : " << elapsed << " s  ("
              << static_cast<double>(step) / elapsed << " steps/s)\n";

    if (benchmark_mode) {
        if (!benchmark_wall_started ||
            advance_timings.samples !=
                static_cast<std::size_t>(rc.benchmark_steps)) {
            std::cerr << "[ERROR] Benchmark ended before collecting the requested "
                      << rc.benchmark_steps << " measured steps; collected "
                      << advance_timings.samples << ".\n";
            free_gpu_workspace(ws);
            return 1;
        }

        const double benchmark_wall_ms =
            std::chrono::duration<double, std::milli>(
                wall_end - benchmark_wall_start).count();
        const double samples = static_cast<double>(advance_timings.samples);
        const std::uint64_t state_hash = interior_state_hash(Uold);

        std::cout << std::fixed << std::setprecision(9)
                  << "[TUNING] measured_steps=" << advance_timings.samples
                  << " wall_ms=" << benchmark_wall_ms
                  << " ms_per_step=" << benchmark_wall_ms / samples
                  << " advance_x_ms_per_step=" << advance_timings.x_ms / samples
                  << " advance_y_ms_per_step=" << advance_timings.y_ms / samples
                  << "\n";
        std::cout << "TUNING_CSV,"
                  << advance_timings.samples << ','
                  << benchmark_wall_ms << ','
                  << benchmark_wall_ms / samples << ','
                  << advance_timings.x_ms / samples << ','
                  << advance_timings.y_ms / samples << ','
                  << std::hex << std::setw(16) << std::setfill('0')
                  << state_hash << std::dec << std::setfill(' ') << "\n";
    }

    // For cases without a snapshot schedule, write final state with legacy tag.
    if (rc.write_out && !has_snaps) {
        write_all_fields(Uold, rc.out_dir,
            rc.case_name + "_gpu_t" +
            std::to_string(static_cast<int>(std::round(cfg.t_end * 100))));
    }

    free_gpu_workspace(ws);
    return 0;
}
