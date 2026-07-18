#pragma once

#include "grid_cpu.hpp"
#include "test_cases.hpp"  // BoundaryConfig, BoundaryType

// ============================================================
// Apply boundary conditions along one coordinate direction.  The split
// solver only needs x ghosts before an x sweep and y ghosts before a y
// sweep, so keeping these operations separate avoids redundant work in the
// timestep loop.
//
// BoundaryType semantics:
//   Periodic     — ghost cells wrap to opposite active boundary
//   Transmissive — ghost cells copy nearest active cell (zero-gradient)
// ============================================================
inline void apply_boundary_x(Grid2D& grid, const BoundaryConfig& bc) {
    const int ng       = grid.ng();
    const int ib       = grid.i_begin();
    const int ie       = grid.i_end();
    const int total_ny = grid.total_ny();

    // ---- Left ----
    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            switch (bc.left) {
                case BoundaryType::Periodic:
                    grid(ib - 1 - g, j) = grid(ie - 1 - g, j);
                    break;
                case BoundaryType::Transmissive:
                    grid(ib - 1 - g, j) = grid(ib, j);
                    break;
            }
        }
    }

    // ---- Right ----
    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            switch (bc.right) {
                case BoundaryType::Periodic:
                    grid(ie + g, j) = grid(ib + g, j);
                    break;
                case BoundaryType::Transmissive:
                    grid(ie + g, j) = grid(ie - 1, j);
                    break;
            }
        }
    }

}

inline void apply_boundary_y(Grid2D& grid, const BoundaryConfig& bc) {
    const int ng       = grid.ng();
    const int jb       = grid.j_begin();
    const int je       = grid.j_end();
    const int total_nx = grid.total_nx();

    // ---- Bottom ----
    for (int i = 0; i < total_nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            switch (bc.bottom) {
                case BoundaryType::Periodic:
                    grid(i, jb - 1 - g) = grid(i, je - 1 - g);
                    break;
                case BoundaryType::Transmissive:
                    grid(i, jb - 1 - g) = grid(i, jb);
                    break;
            }
        }
    }

    // ---- Top ----
    for (int i = 0; i < total_nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            switch (bc.top) {
                case BoundaryType::Periodic:
                    grid(i, je + g) = grid(i, jb + g);
                    break;
                case BoundaryType::Transmissive:
                    grid(i, je + g) = grid(i, je - 1);
                    break;
            }
        }
    }
}

// Full refresh retained for initialisation and non-split callers.
inline void apply_boundary(Grid2D& grid, const BoundaryConfig& bc) {
    apply_boundary_x(grid, bc);
    apply_boundary_y(grid, bc);
}
