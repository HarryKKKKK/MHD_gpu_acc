#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostics.hpp"
#include "gpu/boundary_gpu.cuh"
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
    f << std::scientific << std::setprecision(8);
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

// ============================================================
// Argument parsing
// ============================================================

struct RunConfig {
    std::string   case_name     = "kelvin_helmholtz";
    int           n_scale       = 1;
    RiemannSolver solver        = RiemannSolver::HLLD;
    std::string   out_dir       = "output";
    bool          write_out     = false;
    int           diag_interval = 0;              // 0 = diagnostics disabled
    std::string   diag_dir      = "diagnostics";
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
        } else if (arg == "--diag-interval" && i + 1 < argc) {
            rc.diag_interval = std::stoi(argv[++i]);
        } else if (arg == "--diag-dir" && i + 1 < argc) {
            rc.diag_dir = argv[++i];
        } else if (arg[0] != '-') {
            rc.n_scale = std::stoi(arg);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
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

    const std::string solver_name =
        (rc.solver == RiemannSolver::HLL)  ? "hll"  :
        (rc.solver == RiemannSolver::HLLC) ? "hllc" :
        (rc.solver == RiemannSolver::HLLD) ? "hlld" : "force";

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

    // Apply BCs on the initial data so ghost cells are consistent.
    apply_boundary_gpu(Uold, cfg.bc);

    if (rc.write_out)
        write_all_fields(Uold, rc.out_dir, rc.case_name + "_gpu_t0");

    // ---- Diagnostic CSV (--diag-interval N, N=0 disables) ----
    const bool    diag_on = rc.diag_interval > 0;
    std::ofstream diag_file;
    if (diag_on) {
        namespace fs = std::filesystem;
        fs::create_directories(rc.diag_dir);
        const std::string diag_path = rc.diag_dir + "/diag_gpu_" + rc.case_name +
            "_" + solver_name + "_n" + std::to_string(rc.n_scale) + ".csv";
        diag_file.open(diag_path);
        if (!diag_file)
            throw std::runtime_error("Cannot open diagnostic file: " + diag_path);
        diag_file << "arch,case,solver,step,t,dt,ch_glm,max_cf_x,max_cf_y,max_speed,"
                     "max_rho,min_rho,max_p,min_p,max_abs_Bx,max_abs_By,max_abs_Bz,max_abs_psi,"
                     "divB_L1,divB_max,n_floor_triggered\n";
        diag_file << std::scientific << std::setprecision(10);
        std::cout << "  [diag] every " << rc.diag_interval
                  << " steps -> " << diag_path << "\n" << std::flush;
    }

    const bool has_snaps = !cfg.snapshot_times.empty();
    std::size_t snap_idx = 0;

    double t    = 0.0;
    int    step = 0;
    std::string exit_reason = "reached_t_end";

    auto wall_start = std::chrono::steady_clock::now();

    // The whole loop is wrapped in try/catch purely so a diagnostic run can
    // record exit_reason before an uncaught exception (e.g. compute_dt_gpu's
    // "workspace not initialised") escapes main() — the exception is always
    // rethrown afterwards, so crash/exit-code behaviour when --diag-interval
    // is off (or even on) is unchanged.
    try {
    while (t < cfg.t_end) {
        // Determine the next time we must not overshoot:
        // either the next snapshot or t_end, whichever is sooner.
        double t_next = cfg.t_end;
        if (has_snaps && snap_idx < cfg.snapshot_times.size())
            t_next = std::min(t_next, cfg.snapshot_times[snap_idx]);

        double max_speed_raw = 0.0;
        const double dt_raw = compute_dt_gpu(Uold, ws, cfg.cfl, &max_speed_raw);
        const double dt     = std::min(dt_raw, t_next - t);

        // Diagnostic snapshot of the state compute_dt_gpu() just used, taken
        // *before* advance_second_order_gpu() mutates anything (read-only
        // download + scan — does not affect dt/flux computation below).
        const bool do_diag = diag_on && (step % rc.diag_interval == 0);
        diag::Snapshot snap;
        if (do_diag) {
            std::vector<Conserved> host_vec;
            Uold.download_to_aos(host_vec);
            const int total_nx = Uold.total_nx();
            auto accessor = [&](int i, int j) -> Conserved {
                return host_vec[static_cast<std::size_t>(j) * total_nx + i];
            };
            snap = diag::compute_snapshot(
                accessor,
                Uold.i_begin(), Uold.i_end(), Uold.j_begin(), Uold.j_end(),
                Uold.dx(), Uold.dy());
        }

        // Guard against numerical blowup: dt should never be zero or non-finite.
        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "[ERROR] dt=" << dt << " at step=" << step
                      << " t=" << t << " (dt_raw=" << dt_raw << "). Aborting.\n";
            exit_reason = "dt_invalid";
            break;
        }
        // Warn if dt is suspiciously tiny relative to t_end (signal speed blowup).
        // A dt more than 8 orders of magnitude below t_end/N_expected is pathological.
        const double dt_floor = 1.0e-8 * cfg.t_end;
        if (dt < dt_floor) {
            std::cerr << "[WARN]  dt=" << std::scientific << dt
                      << " (floor=" << dt_floor << ") at step=" << step
                      << " t=" << t << " — numerical blowup, stopping.\n";
            exit_reason = "dt_floor_breach";
            break;
        }

        if (diag_on) reset_floor_trigger_count_gpu();
        advance_second_order_gpu(Uold, Utmp, Unew, ws, dt, rc.solver, cfg.bc);
        const unsigned long long n_floor = diag_on ? read_floor_trigger_count_gpu() : 0ULL;

        if (do_diag) {
            diag_file << "gpu," << rc.case_name << ',' << solver_name << ','
                      << step << ',' << t << ',' << dt << ',' << phys::ch_glm << ','
                      << snap.max_cf_x << ',' << snap.max_cf_y << ',' << max_speed_raw << ','
                      << snap.max_rho << ',' << snap.min_rho << ','
                      << snap.max_p   << ',' << snap.min_p   << ','
                      << snap.max_abs_Bx << ',' << snap.max_abs_By << ',' << snap.max_abs_Bz << ','
                      << snap.max_abs_psi << ','
                      << snap.divB_L1 << ',' << snap.divB_max << ','
                      << n_floor << '\n';
        }

        Uold.swap(Unew);
        t    += dt;
        step += 1;

        // Periodic progress output so the Slurm log is never silent for long.
        if (step % 500 == 0) {
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
        if (has_snaps) {
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
    } catch (const std::exception& e) {
        exit_reason = std::string("exception:") + e.what();
        if (diag_on) {
            diag_file << "# exit_summary: final_step=" << step
                      << ",final_t=" << t
                      << ",t_end_target=" << cfg.t_end
                      << ",exit_reason=" << exit_reason << "\n";
            diag_file.flush();
        }
        throw;
    }

    if (diag_on) {
        diag_file << "# exit_summary: final_step=" << step
                  << ",final_t=" << t
                  << ",t_end_target=" << cfg.t_end
                  << ",exit_reason=" << exit_reason << "\n";
    }

    auto wall_end = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << "[GPU] Total steps= " << step << "\n";
    std::cout << "  Elapsed : " << elapsed << " s  ("
              << static_cast<double>(step) / elapsed << " steps/s)\n";

    // For cases without a snapshot schedule, write final state with legacy tag.
    if (rc.write_out && !has_snaps) {
        write_all_fields(Uold, rc.out_dir,
            rc.case_name + "_gpu_t" +
            std::to_string(static_cast<int>(std::round(cfg.t_end * 100))));
    }

    free_gpu_workspace(ws);
    return 0;
}
