#pragma once

#include <cmath>

#include "gpu/boundary3d_gpu.cuh"
#include "physics.hpp"

// Identical physical configuration to the moderate Athena-style blast on
// the 3D_task branch.
namespace blast3d {

inline constexpr double gamma = 5.0 / 3.0;
inline constexpr double rho = 1.0;
inline constexpr double p_inner = 10.0;
inline constexpr double p_outer = 0.1;
inline constexpr double r_inner = 0.09;
inline constexpr double r_outer = 0.10;
inline constexpr double x_min = -0.5;
inline constexpr double x_max = 0.5;
inline constexpr double t_end = 0.10;
inline constexpr double recommended_cfl = 0.20;

inline double magnetic_field_x() {
    return 1.0 / std::sqrt(2.0);
}

inline double magnetic_field_y() {
    return 1.0 / std::sqrt(2.0);
}

inline double pressure(double r) {
    if (r <= r_inner) return p_inner;
    if (r >= r_outer) return p_outer;
    const double f = (r_outer - r) / (r_outer - r_inner);
    return p_outer + f * (p_inner - p_outer);
}

inline Conserved initial_state(double x, double y, double z) {
    const double r = std::sqrt(x*x + y*y + z*z);
    return phys::prim_to_cons(
        Primitive(rho, 0.0, 0.0, 0.0,
                  magnetic_field_x(), magnetic_field_y(), 0.0,
                  pressure(r), 0.0));
}

inline BoundaryConfig3D boundary_conditions() {
    return BoundaryConfig3D{
        BoundaryType::Periodic, BoundaryType::Periodic,
        BoundaryType::Periodic, BoundaryType::Periodic,
        BoundaryType::Periodic, BoundaryType::Periodic};
}

} // namespace blast3d
