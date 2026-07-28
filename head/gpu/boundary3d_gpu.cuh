#pragma once

#include "gpu/grid3d_gpu.cuh"
#include "test_cases.hpp"

struct BoundaryConfig3D {
    BoundaryType left = BoundaryType::Periodic;
    BoundaryType right = BoundaryType::Periodic;
    BoundaryType bottom = BoundaryType::Periodic;
    BoundaryType top = BoundaryType::Periodic;
    BoundaryType front = BoundaryType::Periodic;
    BoundaryType back = BoundaryType::Periodic;
};

void apply_boundary_gpu_3d(Grid3DGPU& grid, const BoundaryConfig3D& bc);
