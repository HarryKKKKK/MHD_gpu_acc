#pragma once

#include <cmath>

#include "cpu/boundary3d_cpu.hpp"
#include "physics.hpp"

// Compressible ideal-MHD counterpart of the Insulating Magnetic
// Taylor-Green (IMTG) initial condition from Pouquet et al.,
// arXiv:0906.1384, Eqs. (3) and (5)-(7).
//
// The paper evolves incompressible, viscous-resistive MHD with a
// pseudo-spectral solver. This project instead evolves the same normalized,
// divergence-free velocity and magnetic fields with its existing ideal
// compressible GLM-MHD finite-volume solver. A uniform thermal pressure keeps
// the initial flow at low Mach number.
namespace imtg3d {

inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double x_min = 0.0;
inline constexpr double x_max = 2.0*pi;
inline constexpr double rho = 1.0;
inline constexpr double gamma = 5.0/3.0;
inline constexpr double thermal_pressure = 10.0;
inline constexpr double velocity_amplitude = 1.0;
inline constexpr double magnetic_amplitude =
    0.577350269189625764509148780501957456; // 1/sqrt(3)
inline constexpr double t_end = 2.0;
inline constexpr double recommended_cfl = 0.20;

inline Conserved initial_state(double x, double y, double z) {
    const double sx=std::sin(x), cx=std::cos(x);
    const double sy=std::sin(y), cy=std::cos(y);
    const double sz=std::sin(z), cz=std::cos(z);

    const double u= velocity_amplitude*sx*cy*cz;
    const double v=-velocity_amplitude*cx*sy*cz;
    const double w=0.0;
    const double bx= magnetic_amplitude*cx*sy*sz;
    const double by= magnetic_amplitude*sx*cy*sz;
    const double bz=-2.0*magnetic_amplitude*sx*sy*cz;
    return phys::prim_to_cons(
        Primitive(rho,u,v,w,bx,by,bz,thermal_pressure,0.0));
}

inline BoundaryConfig3D boundary_conditions() {
    BoundaryConfig3D bc;
    bc.x_min = bc.x_max = BoundaryType::Periodic;
    bc.y_min = bc.y_max = BoundaryType::Periodic;
    bc.z_min = bc.z_max = BoundaryType::Periodic;
    return bc;
}

} // namespace imtg3d
