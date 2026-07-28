#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "gpu/solver3d_gpu.cuh"
#include "physics.hpp"
#include "test_cases.hpp"

namespace {

struct RunConfig {
    std::string case_name = "orszag_tang";
    int n_scale = 1;
    int resolution = 0;
    int max_steps = 0;
    RiemannSolver solver = RiemannSolver::HLLD;
    bool write_output = false;
    std::string out_dir = "output";
};

RunConfig parse_args(int argc, char** argv) {
    RunConfig rc;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            rc.case_name = argv[++i];
        } else if (arg == "--solver" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "hll") rc.solver = RiemannSolver::HLL;
            else if (value == "hllc") rc.solver = RiemannSolver::HLLC;
            else if (value == "hlld") rc.solver = RiemannSolver::HLLD;
            else if (value == "force") rc.solver = RiemannSolver::FORCE;
            else throw std::runtime_error("Unknown solver: " + value);
        } else if (arg == "--resolution" && i + 1 < argc) {
            rc.resolution = std::stoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            rc.max_steps = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            rc.out_dir = argv[++i];
            rc.write_output = true;
        } else if (arg == "--output") {
            rc.write_output = true;
        } else if (arg == "--no-out") {
            rc.write_output = false;
        } else if (!arg.empty() && arg[0] != '-') {
            rc.n_scale = std::stoi(arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (rc.n_scale < 1 || rc.resolution < 0 || rc.max_steps < 0)
        throw std::runtime_error("n, resolution, and max-steps must be non-negative.");
    if (rc.case_name != "orszag_tang" && rc.case_name != "rotor")
        throw std::runtime_error(
            "3D baseline supports exactly: orszag_tang and rotor.");
    return rc;
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

__global__ void extrude_plane_kernel(Grid3DGPUView grid,
                                     const Conserved* plane) {
    const std::size_t p =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t plane_cells =
        static_cast<std::size_t>(grid.total_nx()) * grid.total_ny();
    const std::size_t total = plane_cells * grid.total_nz();
    if (p < total) grid.cells[p] = plane[p % plane_cells];
}

std::vector<Conserved> make_initial_plane(const CaseConfig& cfg,
                                          CaseId case_id) {
    const int tx = cfg.nx + 2 * cfg.ng;
    const int ty = cfg.ny + 2 * cfg.ng;
    const double dx = (cfg.x_max - cfg.x_min) / cfg.nx;
    const double dy = (cfg.y_max - cfg.y_min) / cfg.ny;
    std::vector<Conserved> plane(static_cast<std::size_t>(tx) * ty);
    phys::gamma = cfg.gamma;
    for (int j = 0; j < ty; ++j) {
        const double y = cfg.y_min + (j - cfg.ng + 0.5) * dy;
        for (int i = 0; i < tx; ++i) {
            const double x = cfg.x_min + (i - cfg.ng + 0.5) * dx;
            plane[static_cast<std::size_t>(j) * tx + i] =
                initial_state_at(case_id, x, y);
        }
    }
    return plane;
}

void upload_extruded(Grid3DGPU& grid,
                     const std::vector<Conserved>& plane) {
    Conserved* device_plane = nullptr;
    const std::size_t bytes = plane.size() * sizeof(Conserved);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_plane), bytes);
    if (err != cudaSuccess)
        throw std::runtime_error("Could not allocate initial 2D device plane.");
    err = cudaMemcpy(device_plane, plane.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_plane);
        throw std::runtime_error("Could not upload initial 2D plane.");
    }
    constexpr int threads = 256;
    const std::size_t cells = grid.num_cells();
    const int blocks = static_cast<int>((cells + threads - 1) / threads);
    extrude_plane_kernel<<<blocks, threads>>>(make_view(grid), device_plane);
    err = cudaGetLastError();
    cudaFree(device_plane);
    if (err != cudaSuccess)
        throw std::runtime_error("Could not launch initial extrusion kernel.");
}

void check_memory_fit(int nx, int ny, int nz, int ng) {
    std::size_t free_bytes = 0, total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed.");
    const std::size_t total_cells =
        static_cast<std::size_t>(nx + 2 * ng) *
        static_cast<std::size_t>(ny + 2 * ng) *
        static_cast<std::size_t>(nz + 2 * ng);
    const std::size_t interior_cells =
        static_cast<std::size_t>(nx) * ny * nz;
    // Four state grids, per-cell CFL buffer, plus a conservative 10% margin
    // for CUB and CUDA runtime allocations.
    const long double raw =
        4.0L * total_cells * sizeof(Conserved) +
        static_cast<long double>(interior_cells) * sizeof(double);
    const std::size_t required =
        static_cast<std::size_t>(std::ceil(raw * 1.10L));
    std::cout << "[GPU3D] estimated_device_bytes: " << required << "\n";
    std::cout << "[GPU3D] free_device_bytes: " << free_bytes << "\n";
    if (required > static_cast<std::size_t>(0.90L * free_bytes))
        throw std::runtime_error(
            "Requested 3D grid does not fit safely in current free GPU memory. "
            "Use --resolution N or a smaller scale.");
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
    try {
        const RunConfig rc = parse_args(argc, argv);
        const CaseId case_id = parse_case_id(rc.case_name);
        CaseConfig cfg = get_n_case_config(case_id, rc.n_scale);
        if (rc.resolution > 0) {
            cfg.nx = rc.resolution;
            cfg.ny = rc.resolution;
        }
        const int nz = cfg.nx;
        const double z_min = cfg.x_min;
        const double z_max = cfg.x_max;
        const BoundaryConfig3D bc{
            cfg.bc.left, cfg.bc.right, cfg.bc.bottom, cfg.bc.top,
            BoundaryType::Periodic, BoundaryType::Periodic};

        std::cout << "=== Unoptimised 3D MHD GPU Baseline ===\n"
                  << "  Case   : " << rc.case_name << "\n"
                  << "  Solver : " << solver_name(rc.solver) << "\n"
                  << "  n      : " << rc.n_scale << "\n"
                  << "[GPU3D] nx: " << cfg.nx << "\n"
                  << "[GPU3D] ny: " << cfg.ny << "\n"
                  << "[GPU3D] nz: " << nz << "\n"
                  << "[GPU3D] total_cells: "
                  << static_cast<std::size_t>(cfg.nx) * cfg.ny * nz << "\n";
        check_memory_fit(cfg.nx, cfg.ny, nz, cfg.ng);

        Grid3DGPU old(cfg.nx, cfg.ny, nz, cfg.ng,
                      cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
                      z_min, z_max);
        Grid3DGPU after_x(cfg.nx, cfg.ny, nz, cfg.ng,
                          cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
                          z_min, z_max);
        Grid3DGPU after_y(cfg.nx, cfg.ny, nz, cfg.ng,
                          cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
                          z_min, z_max);
        Grid3DGPU next(cfg.nx, cfg.ny, nz, cfg.ng,
                       cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
                       z_min, z_max);

        const std::vector<Conserved> plane = make_initial_plane(cfg, case_id);
        upload_extruded(old, plane);
        apply_boundary_gpu_3d(old, bc);
        set_gpu3d_physics_gamma(cfg.gamma);
        set_gpu3d_physics_ch(0.0);
        GpuWorkspace3D ws;
        init_gpu_workspace_3d(ws, old);

        cudaDeviceSynchronize();
        const auto wall_start = std::chrono::steady_clock::now();
        cudaEvent_t event_start, event_end;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_end);
        cudaEventRecord(event_start);

        double time = 0.0;
        int steps = 0;
        while (time < cfg.t_end &&
               (rc.max_steps == 0 || steps < rc.max_steps)) {
            const double raw_dt = compute_dt_gpu_3d(old, ws, cfg.cfl);
            const double dt = std::min(raw_dt, cfg.t_end - time);
            if (!std::isfinite(dt) || dt <= 0.0)
                throw std::runtime_error("Non-positive or non-finite dt.");
            advance_gpu_3d(old, after_x, after_y, next, ws,
                           dt, rc.solver, bc);
            old.swap(next);
            time += dt;
            ++steps;
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
                  << "[GPU3D] Wall elapsed= " << wall_seconds << " s\n";

        if (rc.write_output)
            write_center_slice(old, rc.out_dir + "/" + rc.case_name +
                               "_gpu3d_baseline_center.csv");

        cudaEventDestroy(event_start);
        cudaEventDestroy(event_end);
        free_gpu_workspace_3d(ws);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
