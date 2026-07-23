#pragma once

#include "cpu/grid3d_cpu.hpp"
#include "test_cases.hpp"

struct BoundaryConfig3D {
    BoundaryType x_min = BoundaryType::Transmissive;
    BoundaryType x_max = BoundaryType::Transmissive;
    BoundaryType y_min = BoundaryType::Transmissive;
    BoundaryType y_max = BoundaryType::Transmissive;
    BoundaryType z_min = BoundaryType::Transmissive;
    BoundaryType z_max = BoundaryType::Transmissive;
};

inline int ghost_source(int lower_ghost, int g, int begin, int end,
                        BoundaryType type) {
    if (lower_ghost)
        return type == BoundaryType::Periodic ? end - 1 - g : begin;
    return type == BoundaryType::Periodic ? begin + g : end - 1;
}

inline void apply_boundary_x(Grid3D& q, const BoundaryConfig3D& bc) {
    for (int k=0; k<q.total_nz(); ++k) for (int j=0; j<q.total_ny(); ++j)
        for (int g=0; g<q.ng(); ++g) {
            q(q.i_begin()-1-g,j,k) =
                q(ghost_source(1,g,q.i_begin(),q.i_end(),bc.x_min),j,k);
            q(q.i_end()+g,j,k) =
                q(ghost_source(0,g,q.i_begin(),q.i_end(),bc.x_max),j,k);
        }
}

inline void apply_boundary_y(Grid3D& q, const BoundaryConfig3D& bc) {
    for (int k=0; k<q.total_nz(); ++k) for (int i=0; i<q.total_nx(); ++i)
        for (int g=0; g<q.ng(); ++g) {
            q(i,q.j_begin()-1-g,k) =
                q(i,ghost_source(1,g,q.j_begin(),q.j_end(),bc.y_min),k);
            q(i,q.j_end()+g,k) =
                q(i,ghost_source(0,g,q.j_begin(),q.j_end(),bc.y_max),k);
        }
}

inline void apply_boundary_z(Grid3D& q, const BoundaryConfig3D& bc) {
    for (int j=0; j<q.total_ny(); ++j) for (int i=0; i<q.total_nx(); ++i)
        for (int g=0; g<q.ng(); ++g) {
            q(i,j,q.k_begin()-1-g) =
                q(i,j,ghost_source(1,g,q.k_begin(),q.k_end(),bc.z_min));
            q(i,j,q.k_end()+g) =
                q(i,j,ghost_source(0,g,q.k_begin(),q.k_end(),bc.z_max));
        }
}

inline void apply_boundary(Grid3D& q, const BoundaryConfig3D& bc) {
    apply_boundary_x(q, bc);
    apply_boundary_y(q, bc);
    apply_boundary_z(q, bc);
}

