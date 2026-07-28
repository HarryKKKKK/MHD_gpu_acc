#pragma once

#include <cmath>

#include "gpu/boundary3d_gpu.cuh"
#include "physics.hpp"

// Ms0.2_Ma1 weakly-compressible insulating magnetic Taylor-Green case.
// Constants and formulae are identical to the 3D_task branch.
namespace imtg3d {

inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double x_min = -0.5;
inline constexpr double x_max = 0.5;
inline constexpr double length_scale = 1.0 / (2.0 * pi);
inline constexpr int reference_resolution = 1024;
inline constexpr double reference_mach_s = 0.2;
inline constexpr double reference_mach_a = 1.0;
inline constexpr double reference_pressure = 1.0;
inline constexpr double reference_density = 1.0;
inline constexpr double gamma = 5.0 / 3.0;
inline constexpr double sound_speed =
    1.290994448735805628393088466594133;
inline constexpr double velocity_amplitude =
    2.0 * reference_mach_s * sound_speed;
inline constexpr double magnetic_amplitude =
    velocity_amplitude / 1.732050807568877293527446341505872;
inline constexpr double dynamical_time =
    pi * length_scale / velocity_amplitude;
inline constexpr double t_end = 6.0 * dynamical_time;
inline constexpr double recommended_cfl = 0.20;

inline Conserved initial_state(double x, double y, double z) {
    const double X = x / length_scale;
    const double Y = y / length_scale;
    const double Z = z / length_scale;
    const double sx = std::sin(X), cx = std::cos(X);
    const double sy = std::sin(Y), cy = std::cos(Y);
    const double sz = std::sin(Z), cz = std::cos(Z);

    const double u = velocity_amplitude * sx * cy * cz;
    const double v = -velocity_amplitude * cx * sy * cz;
    const double w = 0.0;
    const double bx = magnetic_amplitude * cx * sy * sz;
    const double by = magnetic_amplitude * sx * cy * sz;
    const double bz = -2.0 * magnetic_amplitude * sx * sy * cz;
    const double pressure =
        reference_pressure
        + (reference_density * velocity_amplitude * velocity_amplitude / 16.0)
          * (std::cos(2.0 * X) + std::cos(2.0 * Y))
          * (std::cos(2.0 * Z) + 2.0);
    const double rho =
        pressure * reference_density / reference_pressure;
    return phys::prim_to_cons(
        Primitive(rho, u, v, w, bx, by, bz, pressure, 0.0));
}

inline BoundaryConfig3D boundary_conditions() {
    return BoundaryConfig3D{
        BoundaryType::Periodic, BoundaryType::Periodic,
        BoundaryType::Periodic, BoundaryType::Periodic,
        BoundaryType::Periodic, BoundaryType::Periodic};
}

} // namespace imtg3d
