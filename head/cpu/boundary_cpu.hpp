#pragma once

#include "grid_cpu.hpp"
#include "test_cases.hpp"  // BoundaryConfig, BoundaryType

// ============================================================
// Copy ghost cells from src to dst (all sides).
// Call this before apply_boundary when Dirichlet BCs are in use,
// so Dirichlet ghost cells carry forward their previous-step values
// rather than the zero-initialized Conserved{} of a fresh grid.
// ============================================================
inline void copy_ghost_cells(const Grid2D& src, Grid2D& dst) {
    const int ng      = src.ng();
    const int ib      = src.i_begin();
    const int ie      = src.i_end();
    const int jb      = src.j_begin();
    const int je      = src.j_end();
    const int total_nx = src.total_nx();
    const int total_ny = src.total_ny();

    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            dst(ib - 1 - g, j) = src(ib - 1 - g, j);
            dst(ie     + g, j) = src(ie     + g, j);
        }
    }
    for (int i = ib; i < ie; ++i) {
        for (int g = 0; g < ng; ++g) {
            dst(i, jb - 1 - g) = src(i, jb - 1 - g);
            dst(i, je     + g) = src(i, je     + g);
        }
    }
}

// ============================================================
// Apply boundary conditions to all four sides of the grid.
//
// BoundaryType semantics:
//   Periodic     — ghost cells wrap to opposite active boundary
//   Transmissive — ghost cells copy nearest active cell (zero-gradient)
//   Reflecting   — normal velocity and normal B are negated
//   Dirichlet    — no-op: caller is responsible for setting ghost
//                  cells (typically via copy_ghost_cells from a
//                  reference/previous-step grid before this call)
// ============================================================
inline void apply_boundary(Grid2D& grid, const BoundaryConfig& bc) {
    const int ng       = grid.ng();
    const int ib       = grid.i_begin();
    const int ie       = grid.i_end();
    const int jb       = grid.j_begin();
    const int je       = grid.j_end();
    const int total_nx = grid.total_nx();
    const int total_ny = grid.total_ny();

    // ---- Left ----
    if (bc.left != BoundaryType::Dirichlet) {
        for (int j = 0; j < total_ny; ++j) {
            for (int g = 0; g < ng; ++g) {
                switch (bc.left) {
                    case BoundaryType::Periodic:
                        grid(ib - 1 - g, j) = grid(ie - 1 - g, j);
                        break;
                    case BoundaryType::Transmissive:
                        grid(ib - 1 - g, j) = grid(ib, j);
                        break;
                    case BoundaryType::Reflecting: {
                        Conserved m = grid(ib + g, j);
                        m.rhou = -m.rhou;
                        m.Bx   = -m.Bx;
                        grid(ib - 1 - g, j) = m;
                        break;
                    }
                    default: break;
                }
            }
        }
    }

    // ---- Right ----
    if (bc.right != BoundaryType::Dirichlet) {
        for (int j = 0; j < total_ny; ++j) {
            for (int g = 0; g < ng; ++g) {
                switch (bc.right) {
                    case BoundaryType::Periodic:
                        grid(ie + g, j) = grid(ib + g, j);
                        break;
                    case BoundaryType::Transmissive:
                        grid(ie + g, j) = grid(ie - 1, j);
                        break;
                    case BoundaryType::Reflecting: {
                        Conserved m = grid(ie - 1 - g, j);
                        m.rhou = -m.rhou;
                        m.Bx   = -m.Bx;
                        grid(ie + g, j) = m;
                        break;
                    }
                    default: break;
                }
            }
        }
    }

    // ---- Bottom ----
    if (bc.bottom != BoundaryType::Dirichlet) {
        for (int i = 0; i < total_nx; ++i) {
            for (int g = 0; g < ng; ++g) {
                switch (bc.bottom) {
                    case BoundaryType::Periodic:
                        grid(i, jb - 1 - g) = grid(i, je - 1 - g);
                        break;
                    case BoundaryType::Transmissive:
                        grid(i, jb - 1 - g) = grid(i, jb);
                        break;
                    case BoundaryType::Reflecting: {
                        Conserved m = grid(i, jb + g);
                        m.rhov = -m.rhov;
                        m.By   = -m.By;
                        grid(i, jb - 1 - g) = m;
                        break;
                    }
                    default: break;
                }
            }
        }
    }

    // ---- Top ----
    if (bc.top != BoundaryType::Dirichlet) {
        for (int i = 0; i < total_nx; ++i) {
            for (int g = 0; g < ng; ++g) {
                switch (bc.top) {
                    case BoundaryType::Periodic:
                        grid(i, je + g) = grid(i, jb + g);
                        break;
                    case BoundaryType::Transmissive:
                        grid(i, je + g) = grid(i, je - 1);
                        break;
                    case BoundaryType::Reflecting: {
                        Conserved m = grid(i, je - 1 - g);
                        m.rhov = -m.rhov;
                        m.By   = -m.By;
                        grid(i, je + g) = m;
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}
