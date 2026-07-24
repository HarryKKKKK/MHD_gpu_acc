#pragma once

#include <cmath>

#include "cpu/boundary3d_cpu.hpp"
#include "physics.hpp"

// Three-dimensional magnetized blast from:
// Derigs et al., J. Comput. Phys. 317 (2016), 223-256, Sec. 5.6.
// DOI: 10.1016/j.jcp.2016.04.048
namespace blast3d {

inline constexpr double gamma = 1.4;
inline constexpr double rho = 1.0;
inline constexpr double p_inner = 1000.0;
inline constexpr double p_outer = 0.1;
inline constexpr double r_inner = 0.09;
inline constexpr double r_outer = 0.10;
inline constexpr double x_min = -0.5;
inline constexpr double x_max = 0.5;
inline constexpr double t_end = 0.01;
inline constexpr double recommended_cfl = 0.20;
inline constexpr double pi = 3.141592653589793238462643383279502884;

inline double magnetic_field_x() {
    return 100.0 / std::sqrt(4.0 * pi);
}

inline double pressure(double r) {
    if (r <= r_inner) return p_inner;
    if (r >= r_outer) return p_outer;
    const double f = (r_outer-r)/(r_outer-r_inner);
    return p_outer + f*(p_inner-p_outer);
}

inline Conserved initial_state(double x, double y, double z) {
    const double r = std::sqrt(x*x+y*y+z*z);
    return phys::prim_to_cons(
        Primitive(rho, 0.0, 0.0, 0.0,
                  magnetic_field_x(), 0.0, 0.0, pressure(r), 0.0));
}

inline BoundaryConfig3D boundary_conditions() {
    BoundaryConfig3D bc;
    bc.x_min = bc.x_max = BoundaryType::Periodic;
    bc.y_min = bc.y_max = BoundaryType::Periodic;
    bc.z_min = bc.z_max = BoundaryType::Periodic;
    return bc;
}

} // namespace blast3d
