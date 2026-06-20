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
    std::string   case_name = "kelvin_helmholtz";
    int           n_scale   = 1;
    int           order     = 2;
    RiemannSolver solver    = RiemannSolver::HLLD;
    std::string   out_dir   = "output";
    bool          write_out = false;
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
            else if (s == "hlld")  rc.solver = RiemannSolver::HLLD;
            else if (s == "force") rc.solver = RiemannSolver::FORCE;
            else throw std::runtime_error("Unknown solver: " + s);
        } else if (arg == "--order" && i + 1 < argc) {
            rc.order = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            rc.out_dir   = argv[++i];
            rc.write_out = true;
        } else if (arg == "--output") {
            rc.write_out = true;
        } else if (arg == "--no-out") {
            rc.write_out = false;
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

    std::cout << "=== MHD GLM GPU Solver ===\n";
    std::cout << "  Case   : " << rc.case_name << "\n";
    std::cout << "  n      : " << rc.n_scale << "\n";
    std::cout << "  Order  : " << rc.order << "\n";
    std::cout << "  Gamma  : " << cfg.gamma << "\n";
    std::cout << "  Solver : "
              << (rc.solver == RiemannSolver::HLL  ? "HLL"  :
                  rc.solver == RiemannSolver::HLLD ? "HLLD" : "FORCE") << "\n";
    std::cout << "[GPU] nx: "          << cfg.nx            << "\n";
    std::cout << "[GPU] ny: "          << cfg.ny            << "\n";
    std::cout << "[GPU] total_cells: " << (cfg.nx * cfg.ny) << "\n";

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

    const bool has_snaps = !cfg.snapshot_times.empty();
    std::size_t snap_idx = 0;

    double t    = 0.0;
    int    step = 0;

    auto wall_start = std::chrono::steady_clock::now();

    while (t < cfg.t_end) {
        // Determine the next time we must not overshoot:
        // either the next snapshot or t_end, whichever is sooner.
        double t_next = cfg.t_end;
        if (has_snaps && snap_idx < cfg.snapshot_times.size())
            t_next = std::min(t_next, cfg.snapshot_times[snap_idx]);

        const double dt_raw = compute_dt_gpu(Uold, ws, cfg.cfl);
        const double dt     = std::min(dt_raw, t_next - t);

        if (rc.order == 2) {
            advance_second_order_gpu(Uold, Utmp, Unew, ws, dt, rc.solver, cfg.bc);
        } else {
            advance_first_order_gpu(Uold, Unew, dt, rc.solver, cfg.bc);
        }

        Uold.swap(Unew);
        t    += dt;
        step += 1;

        // Write any snapshots whose time we have just reached.
        if (rc.write_out && has_snaps) {
            while (snap_idx < cfg.snapshot_times.size() &&
                   t >= cfg.snapshot_times[snap_idx] - 1e-12) {
                std::cout << "  [snap] " << cfg.snapshot_tags[snap_idx]
                          << "  t_phys=" << std::scientific << std::setprecision(6)
                          << t << " s\n";
                write_all_fields(Uold, rc.out_dir,
                    rc.case_name + "_gpu_" + cfg.snapshot_tags[snap_idx]);
                ++snap_idx;
            }
        }
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
