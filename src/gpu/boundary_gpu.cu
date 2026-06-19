#include "gpu/boundary_gpu.cuh"

// ============================================================
// Device helper: copy all 9 MHD fields from src_idx to dst_idx.
// ============================================================
__device__ inline void copy9(Grid2DGPUView& g, int dst, int src) {
    g.rho[dst]  = g.rho[src];
    g.rhou[dst] = g.rhou[src];
    g.rhov[dst] = g.rhov[src];
    g.rhow[dst] = g.rhow[src];
    g.Bx[dst]   = g.Bx[src];
    g.By[dst]   = g.By[src];
    g.Bz[dst]   = g.Bz[src];
    g.E[dst]    = g.E[src];
    g.psi[dst]  = g.psi[src];
}

// ============================================================
// Left / Right boundary kernel.
// One thread handles one j-row (all ghost layers for that row).
// ============================================================
__global__ void apply_lr_bc_kernel(
    Grid2DGPUView g,
    BoundaryType  left_type,
    BoundaryType  right_type
) {
    // Cover total_ny rows (active + ghost) so corners are filled too
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= g.total_ny()) return;

    const int ng = g.ng;
    const int ib = g.i_begin();
    const int ie = g.i_end();

    for (int k = 0; k < ng; ++k) {
        // ----- Left ghost cell at ib-1-k -----
        const int gL  = g.flat_index(ib - 1 - k, j);

        switch (left_type) {
            case BoundaryType::Periodic: {
                const int src = g.flat_index(ie - 1 - k, j);
                copy9(g, gL, src);
                break;
            }
            case BoundaryType::Transmissive: {
                const int src = g.flat_index(ib, j);
                copy9(g, gL, src);
                break;
            }
            case BoundaryType::Reflecting: {
                const int src = g.flat_index(ib + k, j);
                copy9(g, gL, src);
                g.rhou[gL] = -g.rhou[src];   // negate x-momentum
                g.Bx[gL]   = -g.Bx[src];     // negate normal B
                break;
            }
            default: break;  // Dirichlet: caller manages ghost cells
        }

        // ----- Right ghost cell at ie+k -----
        const int gR  = g.flat_index(ie + k, j);

        switch (right_type) {
            case BoundaryType::Periodic: {
                const int src = g.flat_index(ib + k, j);
                copy9(g, gR, src);
                break;
            }
            case BoundaryType::Transmissive: {
                const int src = g.flat_index(ie - 1, j);
                copy9(g, gR, src);
                break;
            }
            case BoundaryType::Reflecting: {
                const int src = g.flat_index(ie - 1 - k, j);
                copy9(g, gR, src);
                g.rhou[gR] = -g.rhou[src];
                g.Bx[gR]   = -g.Bx[src];
                break;
            }
            default: break;
        }
    }
}

// ============================================================
// Bottom / Top boundary kernel.
// One thread handles one i-column (all ghost layers for that column).
// Covers total_nx columns so the corners (already set by lr_bc) are
// not corrupted (we skip j-ghost rows that fall outside the y domain).
// ============================================================
__global__ void apply_bt_bc_kernel(
    Grid2DGPUView g,
    BoundaryType  bottom_type,
    BoundaryType  top_type
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total_nx()) return;

    const int ng = g.ng;
    const int jb = g.j_begin();
    const int je = g.j_end();

    for (int k = 0; k < ng; ++k) {
        // ----- Bottom ghost cell at jb-1-k -----
        const int gB  = g.flat_index(i, jb - 1 - k);

        switch (bottom_type) {
            case BoundaryType::Periodic: {
                const int src = g.flat_index(i, je - 1 - k);
                copy9(g, gB, src);
                break;
            }
            case BoundaryType::Transmissive: {
                const int src = g.flat_index(i, jb);
                copy9(g, gB, src);
                break;
            }
            case BoundaryType::Reflecting: {
                const int src = g.flat_index(i, jb + k);
                copy9(g, gB, src);
                g.rhov[gB] = -g.rhov[src];   // negate y-momentum
                g.By[gB]   = -g.By[src];     // negate normal B
                break;
            }
            default: break;
        }

        // ----- Top ghost cell at je+k -----
        const int gT  = g.flat_index(i, je + k);

        switch (top_type) {
            case BoundaryType::Periodic: {
                const int src = g.flat_index(i, jb + k);
                copy9(g, gT, src);
                break;
            }
            case BoundaryType::Transmissive: {
                const int src = g.flat_index(i, je - 1);
                copy9(g, gT, src);
                break;
            }
            case BoundaryType::Reflecting: {
                const int src = g.flat_index(i, je - 1 - k);
                copy9(g, gT, src);
                g.rhov[gT] = -g.rhov[src];
                g.By[gT]   = -g.By[src];
                break;
            }
            default: break;
        }
    }
}

// ============================================================
// Host-side dispatcher
// ============================================================
void apply_boundary_gpu(Grid2DGPU& grid, const BoundaryConfig& bc) {
    Grid2DGPUView view = make_view(grid);

    // Left / right — one thread per row (total_ny rows)
    {
        const int threads = 256;
        const int blocks  = (grid.total_ny() + threads - 1) / threads;
        apply_lr_bc_kernel<<<blocks, threads>>>(
            view, bc.left, bc.right);
    }

    // Bottom / top — one thread per column (total_nx columns)
    {
        const int threads = 256;
        const int blocks  = (grid.total_nx() + threads - 1) / threads;
        apply_bt_bc_kernel<<<blocks, threads>>>(
            view, bc.bottom, bc.top);
    }
}
