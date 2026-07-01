// MHD solver — CPU driver
// Implements GLM-MHD following Dedner et al. (2002), J. Comput. Phys. 175, 645-673.
//
// Usage:
//   ./main_cpu [case_name] [--n N] [--solver hll|hllc|hlld|force] [--out output_dir] [--no-out]
//              [--diag-interval N] [--diag-dir dir]
//
// --n N              : weak-scaling factor; scales the base grid by N in each dimension (default 1)
// --diag-interval N  : every N steps, append a diagnostic snapshot row (dt, ch_glm,
//                       max_cf_x/y, field extrema, div(B), floor-trigger count) to
//                       <diag_dir>/diag_cpu_<case>_<solver>_n<n>.csv. 0 (default) disables
//                       this entirely — no extra file, no extra per-step work.
// --diag-dir dir     : directory for the diagnostic CSV (default "diagnostics")
//
// Output: one CSV file per field at t_end (rho, p, Bx, By, Bz, psi, u, v, E)
// written to output_dir (default: "output/").

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

#include "cpu/boundary_cpu.hpp"
#include "cpu/grid_cpu.hpp"
#include "cpu/solver_cpu.hpp"
#include "diagnostics.hpp"
#include "init.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

// ============================================================
// CSV output: one file per field, rows = j (y), cols = i (x)
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

    // Atomic rename: final file is either complete or absent
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
    int           diag_interval   = 0;              // 0 = diagnostics disabled
    std::string   diag_dir        = "diagnostics";
};

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
        } else if (arg == "--no-out") {
            rc.write_out = false;
        } else if (arg == "--diag-interval" && i + 1 < argc) {
            rc.diag_interval = std::stoi(argv[++i]);
        } else if (arg == "--diag-dir" && i + 1 < argc) {
            rc.diag_dir = argv[++i];
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
    RunConfig rc;
    try {
        rc = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    const std::string solver_name =
        (rc.solver == RiemannSolver::HLL)  ? "hll"  :
        (rc.solver == RiemannSolver::HLLC) ? "hllc" :
        (rc.solver == RiemannSolver::HLLD) ? "hlld" : "force";

    std::cout << "=== MHD GLM Solver (Dedner et al. 2002) ===\n";
    std::cout << "  Case      : " << rc.case_name << "\n";
    std::cout << "  Scale (n) : " << rc.n_scale << "\n";
    std::cout << "  Solver    : "
              << (rc.solver == RiemannSolver::HLL  ? "HLL"  :
                  rc.solver == RiemannSolver::HLLC ? "HLLC" :
                  rc.solver == RiemannSolver::HLLD ? "HLLD" : "FORCE") << "\n";

    // ---- Case config ----
    CaseConfig cfg;
    try {
        cfg = get_n_case_config(rc.case_name, rc.n_scale);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "  Grid      : " << cfg.nx << " x " << cfg.ny << "\n";
    std::cout << "  Domain    : ["
              << cfg.x_min << "," << cfg.x_max << "] x ["
              << cfg.y_min << "," << cfg.y_max << "]\n";
    std::cout << "  t_end     : " << cfg.t_end << "\n";
    std::cout << "  gamma     : " << cfg.gamma << "\n";

    // ---- Initialise grid ----
    // phys::gamma set inside make_n_grid via init.cpp
    Grid2D Uold = make_n_grid(rc.case_name, rc.n_scale);
    Grid2D Unew = Uold;   // copy (includes ghost cells from init)
    Grid2D Utmp = Uold;   // scratch for second-order advance

    CpuWorkspace ws;
    ws.init(cfg.nx, cfg.ny);

    // ---- Write initial condition ----
    if (rc.write_out)
        write_all_fields(Uold, rc.out_dir, rc.case_name + "_cpu_t0");

    // ---- Diagnostic CSV (--diag-interval N, N=0 disables) ----
    const bool    diag_on = rc.diag_interval > 0;
    std::ofstream diag_file;
    if (diag_on) {
        std::filesystem::create_directories(rc.diag_dir);
        const std::string diag_path = rc.diag_dir + "/diag_cpu_" + rc.case_name +
            "_" + solver_name + "_n" + std::to_string(rc.n_scale) + ".csv";
        diag_file.open(diag_path);
        if (!diag_file)
            throw std::runtime_error("Cannot open diagnostic file: " + diag_path);
        diag_file << "arch,case,solver,step,t,dt,ch_glm,max_cf_x,max_cf_y,max_speed,"
                     "max_rho,min_rho,max_p,min_p,max_abs_Bx,max_abs_By,max_abs_Bz,max_abs_psi,"
                     "divB_L1,divB_max,n_floor_triggered\n";
        diag_file << std::scientific << std::setprecision(10);
        std::cout << "  [diag] every " << rc.diag_interval
                  << " steps -> " << diag_path << "\n";
    }

    // ---- Time loop ----
    const bool   has_snaps = !cfg.snapshot_times.empty();
    std::size_t  snap_idx  = 0;

    double t     = 0.0;
    int    step  = 0;
    double t_print_next = 0.0;
    std::string exit_reason = "reached_t_end";

    auto wall_start = std::chrono::steady_clock::now();

    std::cout << "\n  step       t         dt        max|B|\n";
    std::cout << "  ----  ----------  ----------  ----------\n";

    // The whole loop is wrapped in try/catch purely so a diagnostic run
    // can record exit_reason before an uncaught exception (e.g. compute_dt's
    // "non-positive maximum wave speed") escapes main() — the exception is
    // always rethrown afterwards, so the crash/exit-code behaviour when
    // --diag-interval is off (or even on) is unchanged.
    try {
    while (t < cfg.t_end) {
        // Clamp dt to the next snapshot (or t_end), whichever comes first
        double t_next = cfg.t_end;
        if (has_snaps && snap_idx < cfg.snapshot_times.size())
            t_next = std::min(t_next, cfg.snapshot_times[snap_idx]);

        double max_speed_raw = 0.0;
        const double dt_raw = compute_dt(Uold, cfg.cfl, &max_speed_raw);
        const double dt     = std::min(dt_raw, t_next - t);

        // Diagnostic snapshot of the state compute_dt() just used, taken
        // *before* advance_second_order() mutates anything (pure read-only
        // scan — does not affect dt/flux computation below).
        const bool do_diag = diag_on && (step % rc.diag_interval == 0);
        diag::Snapshot snap;
        if (do_diag) {
            snap = diag::compute_snapshot(
                [&](int i, int j) -> Conserved { return Uold(i, j); },
                Uold.i_begin(), Uold.i_end(), Uold.j_begin(), Uold.j_end(),
                Uold.dx(), Uold.dy());
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "[ERROR] dt=" << dt << " at step=" << step
                      << " t=" << t << " (dt_raw=" << dt_raw << "). Aborting.\n";
            exit_reason = "dt_invalid";
            break;
        }
        const double dt_floor = 1.0e-8 * cfg.t_end;
        if (dt < dt_floor) {
            std::cerr << "[WARN]  dt=" << std::scientific << dt
                      << " (floor=" << dt_floor << ") at step=" << step
                      << " t=" << t << " — numerical blowup, stopping.\n";
            exit_reason = "dt_floor_breach";
            break;
        }

        if (diag_on) diag::reset_floor_trigger_count();
        advance_second_order(Uold, Utmp, Unew, dt, ws, rc.solver, cfg.bc);
        const long long n_floor = diag_on ? diag::floor_trigger_count : 0;

        if (do_diag) {
            diag_file << "cpu," << rc.case_name << ',' << solver_name << ','
                      << step << ',' << t << ',' << dt << ',' << phys::ch_glm << ','
                      << snap.max_cf_x << ',' << snap.max_cf_y << ',' << max_speed_raw << ','
                      << snap.max_rho << ',' << snap.min_rho << ','
                      << snap.max_p   << ',' << snap.min_p   << ','
                      << snap.max_abs_Bx << ',' << snap.max_abs_By << ',' << snap.max_abs_Bz << ','
                      << snap.max_abs_psi << ','
                      << snap.divB_L1 << ',' << snap.divB_max << ','
                      << n_floor << '\n';
        }

        std::swap(Uold, Unew);
        t    += dt;
        step += 1;

        // Console progress
        if (t >= t_print_next || t >= cfg.t_end) {
            double max_B = 0.0;
            for (int j = Uold.j_begin(); j < Uold.j_end(); ++j)
                for (int i = Uold.i_begin(); i < Uold.i_end(); ++i) {
                    const auto& U = Uold(i, j);
                    max_B = std::max(max_B,
                        std::sqrt(U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz));
                }
            std::cout << "  " << std::setw(4) << step
                      << "  " << std::setw(10) << std::fixed << std::setprecision(5) << t
                      << "  " << std::setw(10) << std::scientific << std::setprecision(3) << dt
                      << "  " << std::setw(10) << max_B << "\n";
            t_print_next = t + rc.print_interval;
        }

        // Write snapshots whose time we have just reached
        if (rc.write_out && has_snaps) {
            while (snap_idx < cfg.snapshot_times.size() &&
                   t >= cfg.snapshot_times[snap_idx] - 1e-12) {
                std::cout << "  [snap] " << cfg.snapshot_tags[snap_idx]
                          << "  t_phys=" << std::scientific << std::setprecision(6)
                          << t << " s\n";
                write_all_fields(Uold, rc.out_dir,
                    rc.case_name + "_cpu_" + cfg.snapshot_tags[snap_idx]);
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
    const double elapsed        = std::chrono::duration<double>(wall_end - wall_start).count();
    const double steps_per_s   = static_cast<double>(step) / elapsed;
    const double Mcell_upd_s   = static_cast<double>(step) * cfg.nx * cfg.ny / elapsed / 1.0e6;

    std::cout << "\n  Done: " << step << " steps in " << elapsed << " s  ("
              << steps_per_s << " steps/s)\n";
    std::cout << "[TIMING]"
              << " n="               << rc.n_scale
              << " nx="              << cfg.nx
              << " ny="              << cfg.ny
              << " steps="           << step
              << " elapsed_s="       << std::fixed << std::setprecision(4) << elapsed
              << " steps_per_s="     << std::fixed << std::setprecision(2) << steps_per_s
              << " Mcell_updates_s=" << std::fixed << std::setprecision(1) << Mcell_upd_s
              << "\n" << std::flush;

    // Write final state only for cases without a snapshot schedule
    if (rc.write_out && !has_snaps) {
        write_all_fields(Uold, rc.out_dir,
            rc.case_name + "_cpu_t" +
            std::to_string(static_cast<int>(std::round(cfg.t_end * 100))));
    }

    return 0;
}
