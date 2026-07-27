#pragma once

#include "gpu/grid_gpu.cuh"
#include "test_cases.hpp"   // BoundaryConfig, BoundaryType

// Directional boundary refreshes used by the dimensionally split solver.
// Handles Periodic and Transmissive types for all five Euler fields.
void apply_boundary_x_gpu(Grid2DGPU& grid, const BoundaryConfig& bc);
void apply_boundary_y_gpu(Grid2DGPU& grid, const BoundaryConfig& bc);

// Full refresh retained for initialisation and non-split callers.
void apply_boundary_gpu(Grid2DGPU& grid, const BoundaryConfig& bc);

// Convenience wrapper: transmissive on all four sides.
inline void apply_transmissive_boundary_gpu(Grid2DGPU& grid) {
    apply_boundary_gpu(grid, BoundaryConfig{});
}
