// MHD solver — CPU driver
// Implements GLM-MHD following Dedner et al. (2002), J. Comput. Phys. 175, 645-673.
//
// Usage:
//   ./main_cpu [case_name] [--order 1|2] [--solver hll|hlld|force] [--out output_dir]
//
// case_name: kelvin_helmholtz
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
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open output file: " + path);

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
}

void write_all_fields(
    const Grid2D&      grid,
    const std::string& dir,
    const std::string& prefix
) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);

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
    std::string  case_name  = "kelvin_helmholtz";
    int          order      = 2;
    RiemannSolver solver    = RiemannSolver::HLLD;
    std::string  out_dir    = "output";
    bool         write_out  = true;
    double       print_interval = 0.1;  // console update every this many t units
};

RunConfig parse_args(int argc, char** argv) {
    RunConfig rc;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--order" && i + 1 < argc) {
            rc.order = std::stoi(argv[++i]);
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

    std::cout << "=== MHD GLM Solver (Dedner et al. 2002) ===\n";
    std::cout << "  Case      : " << rc.case_name << "\n";
    std::cout << "  Order     : " << rc.order << "\n";
    std::cout << "  Solver    : "
              << (rc.solver == RiemannSolver::HLL  ? "HLL"  :
                  rc.solver == RiemannSolver::HLLC ? "HLLC" :
                  rc.solver == RiemannSolver::HLLD ? "HLLD" : "FORCE") << "\n";

    // ---- Case config ----
    CaseConfig cfg;
    try {
        cfg = get_case_config(rc.case_name);
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
    // phys::gamma set inside make_initial_grid via init.cpp
    Grid2D Uold = make_initial_grid(parse_case_id(rc.case_name));
    Grid2D Unew = Uold;   // copy (includes ghost cells from init)
    Grid2D Utmp = Uold;   // scratch for second-order advance

    CpuWorkspace ws;
    if (rc.order == 2) ws.init(cfg.nx, cfg.ny);

    // ---- Write initial condition ----
    if (rc.write_out)
        write_all_fields(Uold, rc.out_dir, rc.case_name + "_t0");

    // ---- Time loop ----
    double t     = 0.0;
    int    step  = 0;
    double t_print_next = 0.0;

    auto wall_start = std::chrono::steady_clock::now();

    std::cout << "\n  step       t         dt        max|B|\n";
    std::cout << "  ----  ----------  ----------  ----------\n";

    while (t < cfg.t_end) {
        const double dt_raw = compute_dt(Uold, cfg.cfl);
        const double dt     = std::min(dt_raw, cfg.t_end - t);

        if (rc.order == 2) {
            advance_second_order(Uold, Utmp, Unew, dt, ws, rc.solver, cfg.bc);
        } else {
            advance_first_order(Uold, Unew, dt, rc.solver, cfg.bc);
        }

        std::swap(Uold, Unew);
        t    += dt;
        step += 1;

        // Console progress
        if (t >= t_print_next || t >= cfg.t_end) {
            // Compute max |B| for diagnostics
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
    }

    auto wall_end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();
    std::cout << "\n  Done: " << step << " steps in " << elapsed << " s  ("
              << static_cast<double>(step) / elapsed << " steps/s)\n";

    // ---- Write final output ----
    if (rc.write_out) {
        write_all_fields(Uold, rc.out_dir,
            rc.case_name + "_t" + std::to_string(static_cast<int>(std::round(cfg.t_end * 100))));
    }

    return 0;
}
