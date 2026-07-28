#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blast3d_case.hpp"
#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "gpu/solver3d_gpu.cuh"
#include "imtg3d_case.hpp"
#include "physics.hpp"

namespace {

enum class Case3D { Blast, IMTG };

struct RunConfig {
    Case3D test_case = Case3D::Blast;
    int resolution = -1;
    int snapshots = -1;
    int max_steps = 0;
    double t_end = -1.0;
    double cfl = -1.0;
    RiemannSolver solver = RiemannSolver::HLLD;
    bool write_output = false;
    std::string out_dir = "output";
};

RunConfig parse_args(int argc, char** argv) {
    RunConfig rc;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value after " + arg);
            return std::string(argv[++i]);
        };
        if (arg == "--case") {
            const std::string name = value();
            if (name == "blast" || name == "blast_athena")
                rc.test_case = Case3D::Blast;
            else if (name == "imtg")
                rc.test_case = Case3D::IMTG;
            else
                throw std::runtime_error(
                    "3D baseline supports exactly: blast and imtg.");
        } else if (arg == "--solver") {
            const std::string name = value();
            if (name == "hll") rc.solver = RiemannSolver::HLL;
            else if (name == "hllc") rc.solver = RiemannSolver::HLLC;
            else if (name == "hlld") rc.solver = RiemannSolver::HLLD;
            else if (name == "force") rc.solver = RiemannSolver::FORCE;
            else throw std::runtime_error("Unknown solver: " + name);
        } else if (arg == "--resolution") {
            rc.resolution = std::stoi(value());
        } else if (arg == "--max-steps") {
            rc.max_steps = std::stoi(value());
        } else if (arg == "--snapshots") {
            rc.snapshots = std::stoi(value());
        } else if (arg == "--t-end") {
            rc.t_end = std::stod(value());
        } else if (arg == "--cfl") {
            rc.cfl = std::stod(value());
        } else if (arg == "--out") {
            rc.out_dir = value();
            rc.write_output = true;
        } else if (arg == "--output") {
            rc.write_output = true;
        } else if (arg == "--no-out") {
            rc.write_output = false;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    const bool imtg = rc.test_case == Case3D::IMTG;
    if (rc.resolution < 0)
        rc.resolution = imtg ? imtg3d::reference_resolution : 128;
    if (rc.snapshots < 0)
        rc.snapshots = imtg ? 12 : 5;
    if (rc.t_end < 0.0)
        rc.t_end = imtg ? imtg3d::t_end : blast3d::t_end;
    if (rc.cfl < 0.0)
        rc.cfl = imtg ? imtg3d::recommended_cfl : blast3d::recommended_cfl;
    if (rc.resolution < 8 || rc.snapshots < 1 || rc.max_steps < 0 ||
        rc.t_end <= 0.0 || rc.cfl <= 0.0)
        throw std::runtime_error(
            "Require resolution >= 8, snapshots >= 1, max-steps >= 0, "
            "t-end > 0, cfl > 0.");
    return rc;
}

const char* case_name(Case3D test_case) {
    return test_case == Case3D::IMTG ? "imtg" : "blast";
}

const char* solver_name(RiemannSolver solver) {
    switch (solver) {
        case RiemannSolver::HLL: return "HLL";
        case RiemannSolver::HLLC: return "HLLC";
        case RiemannSolver::HLLD: return "HLLD";
        case RiemannSolver::FORCE: return "FORCE";
    }
    return "unknown";
}

void check_memory_fit(int n, int ng) {
    std::size_t free_bytes = 0, total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed.");
    const std::size_t extent = static_cast<std::size_t>(n + 2 * ng);
    const std::size_t total_cells = extent * extent * extent;
    const std::size_t interior_cells =
        static_cast<std::size_t>(n) * n * n;
    const long double raw =
        4.0L * total_cells * sizeof(Conserved) +
        static_cast<long double>(interior_cells) * sizeof(double);
    const std::size_t required =
        static_cast<std::size_t>(std::ceil(raw * 1.10L));
    std::cout << "[GPU3D] estimated_device_bytes: " << required << "\n"
              << "[GPU3D] free_device_bytes: " << free_bytes << "\n";
    if (required > static_cast<std::size_t>(0.90L * free_bytes))
        throw std::runtime_error(
            "Requested grid does not fit safely in current free GPU memory. "
            "Choose a smaller --resolution.");
}

std::vector<Conserved> make_initial_volume(Case3D test_case,
                                           int n, int ng,
                                           double lo, double hi) {
    const int extent = n + 2 * ng;
    const std::size_t cells =
        static_cast<std::size_t>(extent) * extent * extent;
    std::vector<Conserved> initial(cells);
    const double d = (hi - lo) / n;
    for (int k = 0; k < extent; ++k) {
        const double z = lo + (k - ng + 0.5) * d;
        for (int j = 0; j < extent; ++j) {
            const double y = lo + (j - ng + 0.5) * d;
            for (int i = 0; i < extent; ++i) {
                const double x = lo + (i - ng + 0.5) * d;
                const std::size_t p =
                    (static_cast<std::size_t>(k) * extent + j) * extent + i;
                initial[p] = test_case == Case3D::IMTG
                    ? imtg3d::initial_state(x, y, z)
                    : blast3d::initial_state(x, y, z);
            }
        }
    }
    return initial;
}

void write_center_slice(const Grid3DGPU& grid,
                        const std::string& path) {
    std::vector<Conserved> host;
    grid.download(host);
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output: " + path);
    out << "i,j,rho,u,v,w,Bx,By,Bz,E,psi,p\n";
    out << std::scientific << std::setprecision(10);
    const int k = grid.k_begin() + grid.nz() / 2;
    const int tx = grid.total_nx();
    const int ty = grid.total_ny();
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& u =
                host[(static_cast<std::size_t>(k) * ty + j) * tx + i];
            const Primitive v = phys::cons_to_prim(u);
            out << i - grid.i_begin() << ',' << j - grid.j_begin() << ','
                << u.rho << ',' << v.u << ',' << v.v << ',' << v.w << ','
                << u.Bx << ',' << u.By << ',' << u.Bz << ',' << u.E << ','
                << u.psi << ',' << v.p << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    GpuWorkspace3D ws;
    try {
        const RunConfig rc = parse_args(argc, argv);
        const bool imtg = rc.test_case == Case3D::IMTG;
        const int n = rc.resolution;
        constexpr int ng = 2;
        const double lo = imtg ? imtg3d::x_min : blast3d::x_min;
        const double hi = imtg ? imtg3d::x_max : blast3d::x_max;
        const double gamma = imtg ? imtg3d::gamma : blast3d::gamma;
        const BoundaryConfig3D bc = imtg
            ? imtg3d::boundary_conditions()
            : blast3d::boundary_conditions();

        std::cout << "=== Unoptimised 3D MHD GPU Baseline ===\n"
                  << "  Case   : " << case_name(rc.test_case) << "\n"
                  << "  Solver : " << solver_name(rc.solver) << "\n"
                  << "  Domain : [" << lo << "," << hi << "]^3\n"
                  << "  Gamma  : " << std::setprecision(17) << gamma << "\n"
                  << "  CFL    : " << rc.cfl << "\n"
                  << "  t_end  : " << rc.t_end << "\n"
                  << "  Targets: " << rc.snapshots << " intervals\n"
                  << "[GPU3D] nx: " << n << "\n"
                  << "[GPU3D] ny: " << n << "\n"
                  << "[GPU3D] nz: " << n << "\n"
                  << "[GPU3D] total_cells: "
                  << static_cast<std::size_t>(n) * n * n << "\n";
        check_memory_fit(n, ng);

        Grid3DGPU old(n, n, n, ng, lo, hi, lo, hi, lo, hi);
        Grid3DGPU after_x(n, n, n, ng, lo, hi, lo, hi, lo, hi);
        Grid3DGPU after_y(n, n, n, ng, lo, hi, lo, hi, lo, hi);
        Grid3DGPU next(n, n, n, ng, lo, hi, lo, hi, lo, hi);

        phys::gamma = gamma;
        const std::vector<Conserved> initial =
            make_initial_volume(rc.test_case, n, ng, lo, hi);
        old.upload(initial);
        apply_boundary_gpu_3d(old, bc);
        set_gpu3d_physics_gamma(gamma);
        set_gpu3d_physics_ch(0.0);
        init_gpu_workspace_3d(ws, old);

        cudaDeviceSynchronize();
        const auto wall_start = std::chrono::steady_clock::now();
        cudaEvent_t event_start, event_end;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_end);
        cudaEventRecord(event_start);

        double time = 0.0;
        int steps = 0;
        int target_index = 1;
        std::vector<double> targets;
        for (int s = 0; s <= rc.snapshots; ++s)
            targets.push_back(
                rc.t_end * static_cast<double>(s) / rc.snapshots);
        while (time < rc.t_end - 1.0e-14 &&
               (rc.max_steps == 0 || steps < rc.max_steps)) {
            const double raw_dt = compute_dt_gpu_3d(old, ws, rc.cfl);
            const double dt = std::min(
                raw_dt, targets[static_cast<std::size_t>(target_index)] - time);
            if (!std::isfinite(dt) || dt <= 0.0)
                throw std::runtime_error("Non-positive or non-finite dt.");
            advance_gpu_3d(old, after_x, after_y, next, ws,
                           dt, rc.solver, bc);
            old.swap(next);
            time += dt;
            ++steps;
            if (time >= targets[static_cast<std::size_t>(target_index)] -
                        1.0e-12 &&
                target_index < rc.snapshots)
                ++target_index;
            if (steps % 100 == 0)
                std::cout << "  [step " << steps << "] t="
                          << std::scientific << time << "\n" << std::flush;
        }

        cudaEventRecord(event_end);
        cudaEventSynchronize(event_end);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, event_start, event_end);
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

        std::cout << "[GPU3D] Total steps= " << steps << "\n"
                  << "[GPU3D] Final time= " << std::setprecision(17) << time << "\n"
                  << "[GPU3D] GPU elapsed= " << gpu_ms / 1000.0 << " s\n"
                  << "[GPU3D] Wall elapsed= " << wall_seconds << " s\n"
                  << "[GPU3D] nx=" << n << " ny=" << n << " nz=" << n
                  << " steps=" << steps << " elapsed_s=" << wall_seconds
                  << "\n";

        if (rc.write_output)
            write_center_slice(old, rc.out_dir + "/" +
                               case_name(rc.test_case) +
                               "_gpu3d_baseline_center.csv");

        cudaEventDestroy(event_start);
        cudaEventDestroy(event_end);
        free_gpu_workspace_3d(ws);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        try {
            free_gpu_workspace_3d(ws);
        } catch (...) {
        }
        return 1;
    }
}
