// MHD solver — pure MPI driver
// Implements GLM-MHD following Dedner et al. (2002), J. Comput. Phys. 175, 645-673.
// Same algorithm/optimisations as scripts/cpu/main_cpu.cpp (OpenMP CPU driver);
// only the parallelisation strategy differs — see head/cpu/solver_mpi.hpp.
//
// Usage:
//   mpirun -np P ./main_mpi [case_name] [N] [--n N] [--solver hll|hlld|force]
//                           [--out output_dir] [--no-out|--timing-only] [--output]
//
// N / --n N : weak-scaling factor; scales the base grid by N in each dimension
//             (default 1). A bare positional integer is accepted as a shorthand
//             for --n (e.g. `main_mpi 2` == `main_mpi --n 2`).
//
// Output: one CSV file per field at t_end (rho, p, Bx, By, Bz, psi, u, v, E),
// gathered onto rank 0 and written to output_dir (default: "output/").

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include "cpu/grid_cpu.hpp"
#include "cpu/solver_mpi.hpp"
#include "init.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

// ============================================================
// CSV output: one file per field, rows = j (y), cols = i (x).
// Identical format to write_field_csv()/write_all_fields() in
// main_cpu.cpp so CPU and MPI runs can be diffed/compared directly;
// duplicated here to keep this driver self-contained.
// ============================================================

template<typename FieldFn>
void write_field_csv(
    const Grid2D&      grid,
    const std::string& path,
    FieldFn            field_fn   // Conserved -> double
) {
    namespace fs = std::filesystem;
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path);
        if (!f) throw std::runtime_error("Cannot open output file: " + tmp_path);

        const int ib = grid.i_begin(), ie = grid.i_end();
        const int jb = grid.j_begin(), je = grid.j_end();

        f << std::scientific << std::setprecision(8);
        for (int j = je - 1; j >= jb; --j) {        // top row first (conventional image layout)
            for (int i = ib; i < ie; ++i) {
                if (i > ib) f << ',';
                f << field_fn(grid(i, j));
            }
            f << '\n';
        }
        if (!f) throw std::runtime_error("Write error: " + tmp_path);
    } // flush + close before rename

    fs::rename(tmp_path, path);
}

void write_all_fields(
    const Grid2D&      grid,
    const std::string& dir,
    const std::string& prefix
) {
    std::filesystem::create_directories(dir);

    const auto path = [&](const char* name) {
        return dir + "/" + prefix + "_" + name + ".csv";
    };

    write_field_csv(grid, path("rho"), [](const Conserved& U){ return U.rho; });
    write_field_csv(grid, path("u"),   [](const Conserved& U){ return U.rhou / U.rho; });
    write_field_csv(grid, path("v"),   [](const Conserved& U){ return U.rhov / U.rho; });
    write_field_csv(grid, path("Bx"),  [](const Conserved& U){ return U.Bx; });
    write_field_csv(grid, path("By"),  [](const Conserved& U){ return U.By; });
    write_field_csv(grid, path("Bz"),  [](const Conserved& U){ return U.Bz; });
    write_field_csv(grid, path("E"),   [](const Conserved& U){ return U.E; });
    write_field_csv(grid, path("psi"), [](const Conserved& U){ return U.psi; });
    write_field_csv(grid, path("p"),   [](const Conserved& U){
        const Primitive V = phys::cons_to_prim(U);
        return V.p;
    });

    std::cout << "  Wrote fields to " << dir << "/" << prefix << "_*.csv\n";
}

// ============================================================
// Argument parsing
// ============================================================

struct RunConfig {
    std::string   case_name       = "kelvin_helmholtz";
    int           n_scale         = 1;
    RiemannSolver solver          = RiemannSolver::HLLD;
    std::string   out_dir         = "output";
    bool          write_out       = true;
    double        print_interval  = 0.1;
};

bool is_all_digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

RunConfig parse_args(int argc, char** argv) {
    RunConfig rc;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            rc.n_scale = std::stoi(argv[++i]);
            if (rc.n_scale < 1) throw std::runtime_error("--n must be >= 1");
        } else if (arg == "--solver" && i + 1 < argc) {
            std::string s = argv[++i];
            if      (s == "hll")   rc.solver = RiemannSolver::HLL;
            else if (s == "hllc")  rc.solver = RiemannSolver::HLLC;
            else if (s == "hlld")  rc.solver = RiemannSolver::HLLD;
            else if (s == "force") rc.solver = RiemannSolver::FORCE;
            else throw std::runtime_error("Unknown solver: " + s);
        } else if (arg == "--out" && i + 1 < argc) {
            rc.out_dir = argv[++i];
        } else if (arg == "--no-out" || arg == "--timing-only") {
            rc.write_out = false;
        } else if (arg == "--output") {
            rc.write_out = true;
        } else if (is_all_digits(arg)) {
            // Bare positional integer: weak-scaling factor N (shorthand for --n N)
            rc.n_scale = std::stoi(arg);
            if (rc.n_scale < 1) throw std::runtime_error("n_scale must be >= 1");
        } else if (arg[0] != '-') {
            rc.case_name = arg;
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
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool is_root = (world_rank == 0);

    int exit_code = 0;
    try {
        RunConfig rc;
        try {
            rc = parse_args(argc, argv);
        } catch (const std::exception& e) {
            if (is_root) std::cerr << "Argument error: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        if (is_root) {
            std::cout << "=== MHD GLM Solver, pure MPI (Dedner et al. 2002) ===\n";
            std::cout << "  Case      : " << rc.case_name << "\n";
            std::cout << "  Scale (n) : " << rc.n_scale << "\n";
            std::cout << "  Solver    : "
                      << (rc.solver == RiemannSolver::HLL  ? "HLL"  :
                          rc.solver == RiemannSolver::HLLC ? "HLLC" :
                          rc.solver == RiemannSolver::HLLD ? "HLLD" : "FORCE") << "\n";
        }

        // ---- Case config (deterministic; computed independently on every rank) ----
        CaseConfig cfg;
        try {
            cfg = get_n_case_config(rc.case_name, rc.n_scale);
        } catch (const std::exception& e) {
            if (is_root) std::cerr << "Error: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // ---- Domain decomposition ----
        MpiDomain dom = mpi_domain_create(MPI_COMM_WORLD, cfg.nx, cfg.ny, cfg.ng, cfg.bc);

        if (is_root) {
            std::cout << "  Grid      : " << cfg.nx << " x " << cfg.ny
                      << "  (ranks: " << dom.dims[0] << " x " << dom.dims[1]
                      << " = " << dom.nprocs << ")\n";
            std::cout << "  Domain    : ["
                      << cfg.x_min << "," << cfg.x_max << "] x ["
                      << cfg.y_min << "," << cfg.y_max << "]\n";
            std::cout << "  t_end     : " << cfg.t_end << "\n";
            std::cout << "  gamma     : " << cfg.gamma << "\n";
        }

        // ---- Initialise local grids ----
        // phys::gamma set inside make_local_grid via init.cpp
        Grid2D Uold = make_local_grid(dom, cfg, rc.case_name);
        Grid2D Unew = Uold;   // copy (includes this rank's ghost cells from init)
        Grid2D Utmp = Uold;   // scratch for second-order advance

        MpiWorkspace ws;
        ws.init(dom.nx_local, dom.ny_local);

        // ---- Write initial condition ----
        if (rc.write_out) {
            Grid2D global0 = gather_global_grid(dom, Uold, cfg);
            if (is_root) write_all_fields(global0, rc.out_dir, rc.case_name + "_mpi_t0");
        }

        // ---- Time loop ----
        const bool   has_snaps = !cfg.snapshot_times.empty();
        std::size_t  snap_idx  = 0;

        double t     = 0.0;
        int    step  = 0;
        double t_print_next = 0.0;

        auto wall_start = std::chrono::steady_clock::now();

        if (is_root) {
            std::cout << "\n  step       t         dt        max|B|\n";
            std::cout << "  ----  ----------  ----------  ----------\n";
        }

        while (t < cfg.t_end) {
            // Clamp dt to the next snapshot (or t_end), whichever comes first.
            // Every rank computes the same t_next/dt deterministically, so no
            // extra synchronisation is needed beyond the Allreduce inside
            // compute_dt_mpi.
            double t_next = cfg.t_end;
            if (has_snaps && snap_idx < cfg.snapshot_times.size())
                t_next = std::min(t_next, cfg.snapshot_times[snap_idx]);

            const double dt_raw = compute_dt_mpi(Uold, cfg.cfl, dom.cart_comm);
            const double dt     = std::min(dt_raw, t_next - t);

            if (!std::isfinite(dt) || dt <= 0.0) {
                if (is_root)
                    std::cerr << "[ERROR] dt=" << dt << " at step=" << step
                              << " t=" << t << " (dt_raw=" << dt_raw << "). Aborting.\n";
                break;
            }
            const double dt_floor = 1.0e-8 * cfg.t_end;
            if (dt < dt_floor) {
                if (is_root)
                    std::cerr << "[WARN]  dt=" << std::scientific << dt
                              << " (floor=" << dt_floor << ") at step=" << step
                              << " t=" << t << " — numerical blowup, stopping.\n";
                break;
            }

            advance_second_order_mpi(Uold, Utmp, Unew, dt, ws, rc.solver, cfg.bc, dom);

            std::swap(Uold, Unew);
            t    += dt;
            step += 1;

            // Console progress (rank 0 only; local maxima reduced across ranks)
            if (t >= t_print_next || t >= cfg.t_end) {
                double local_max_B = 0.0;
                for (int j = Uold.j_begin(); j < Uold.j_end(); ++j)
                    for (int i = Uold.i_begin(); i < Uold.i_end(); ++i) {
                        const auto& U = Uold(i, j);
                        local_max_B = std::max(local_max_B,
                            std::sqrt(U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz));
                    }
                double global_max_B = 0.0;
                MPI_Reduce(&local_max_B, &global_max_B, 1, MPI_DOUBLE, MPI_MAX, 0, dom.cart_comm);

                if (is_root) {
                    std::cout << "  " << std::setw(4) << step
                              << "  " << std::setw(10) << std::fixed << std::setprecision(5) << t
                              << "  " << std::setw(10) << std::scientific << std::setprecision(3) << dt
                              << "  " << std::setw(10) << global_max_B << "\n";
                }
                t_print_next = t + rc.print_interval;
            }

            // Write snapshots whose time we have just reached
            if (has_snaps) {
                while (snap_idx < cfg.snapshot_times.size() &&
                       t >= cfg.snapshot_times[snap_idx] - 1e-12) {
                    if (is_root)
                        std::cout << "  [snap] " << cfg.snapshot_tags[snap_idx]
                                  << "  t_phys=" << std::scientific << std::setprecision(6)
                                  << t << " s\n";
                    if (rc.write_out) {
                        Grid2D global = gather_global_grid(dom, Uold, cfg);
                        if (is_root)
                            write_all_fields(global, rc.out_dir,
                                rc.case_name + "_mpi_" + cfg.snapshot_tags[snap_idx]);
                    }
                    ++snap_idx;
                }
            }
        }

        auto wall_end = std::chrono::steady_clock::now();
        const double elapsed      = std::chrono::duration<double>(wall_end - wall_start).count();
        const double steps_per_s  = static_cast<double>(step) / elapsed;
        const double Mcell_upd_s  = static_cast<double>(step) * cfg.nx * cfg.ny / elapsed / 1.0e6;

        if (is_root) {
            std::cout << "\n  Done: " << step << " steps in " << elapsed << " s  ("
                      << steps_per_s << " steps/s)\n";
            std::cout << "[TIMING]"
                      << " n="               << rc.n_scale
                      << " nx="              << cfg.nx
                      << " ny="              << cfg.ny
                      << " ranks="           << dom.nprocs
                      << " steps="           << step
                      << " elapsed_s="       << std::fixed << std::setprecision(4) << elapsed
                      << " steps_per_s="     << std::fixed << std::setprecision(2) << steps_per_s
                      << " Mcell_updates_s=" << std::fixed << std::setprecision(1) << Mcell_upd_s
                      << "\n" << std::flush;
        }

        // Write final state only for cases without a snapshot schedule
        if (rc.write_out && !has_snaps) {
            Grid2D global_final = gather_global_grid(dom, Uold, cfg);
            if (is_root)
                write_all_fields(global_final, rc.out_dir,
                    rc.case_name + "_mpi_t" +
                    std::to_string(static_cast<int>(std::round(cfg.t_end * 100))));
        }

        mpi_domain_destroy(dom);
    } catch (const std::exception& e) {
        if (is_root) std::cerr << "[FATAL] " << e.what() << "\n";
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
