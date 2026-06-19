#pragma once

#include "gpu/grid_gpu.cuh"
#include "test_cases.hpp"   // BoundaryConfig, BoundaryType

// Apply all four boundary conditions (left/right/bottom/top) to a GPU grid.
// Handles Periodic, Transmissive, Reflecting, and Dirichlet (no-op) types.
// All 9 GLM-MHD fields are updated.
void apply_boundary_gpu(Grid2DGPU& grid, const BoundaryConfig& bc);

// Convenience wrapper: transmissive on all four sides.
inline void apply_transmissive_boundary_gpu(Grid2DGPU& grid) {
    apply_boundary_gpu(grid, BoundaryConfig{});
}
