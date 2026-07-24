#pragma once

#include "cpu/boundary3d_cpu.hpp"
#include "gpu/grid3d_gpu.cuh"

void apply_boundary_x_gpu(Grid3DGPU&,const BoundaryConfig3D&);
void apply_boundary_y_gpu(Grid3DGPU&,const BoundaryConfig3D&);
void apply_boundary_z_gpu(Grid3DGPU&,const BoundaryConfig3D&);
void apply_boundary_gpu(Grid3DGPU&,const BoundaryConfig3D&);

