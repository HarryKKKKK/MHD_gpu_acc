#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blast3d_case.hpp"
#include "cpu/grid3d_cpu.hpp"
#include "cpu/solver3d_cpu.hpp"

namespace {
struct Options {
    int n=128;double t_end=blast3d::t_end,cfl=blast3d::recommended_cfl;
    RiemannSolver solver=RiemannSolver::HLLC;
};
Options parse_args(int argc,char** argv) {
    Options o;
    for(int a=1;a<argc;++a) {
        const std::string s=argv[a];
        auto value=[&]{if(++a>=argc)throw std::runtime_error("missing value after "+s);
                       return std::string(argv[a]);};
        if(s=="--case") {
            if(value()!="blast")throw std::runtime_error("only Euler case 'blast' is available");
        } else if(s=="--resolution")o.n=std::stoi(value());
        else if(s=="--t-end")o.t_end=std::stod(value());
        else if(s=="--cfl")o.cfl=std::stod(value());
        else if(s=="--snapshots")value();
        else if(s=="--no-out"||s=="--timing-only"){}
        else if(s=="--solver") {
            const auto v=value();
            if(v=="hll")o.solver=RiemannSolver::HLL;
            else if(v=="hllc")o.solver=RiemannSolver::HLLC;
            else if(v=="force")o.solver=RiemannSolver::FORCE;
            else throw std::runtime_error("solver must be hll, hllc, or force");
        } else throw std::runtime_error("unknown argument: "+s);
    }
    return o;
}
struct Context {MPI_Comm comm;int rank,ranks;};
void exchange_z(Grid3D& q,void* opaque) {
    auto& c=*static_cast<Context*>(opaque);
    if(c.ranks==1) {
        auto bc=blast3d::boundary_conditions();apply_boundary_z(q,bc);return;
    }
    static_assert(std::is_trivially_copyable<Conserved>::value);
    const std::size_t plane=static_cast<std::size_t>(q.total_nx())*q.total_ny();
    const std::size_t bytes=q.ng()*plane*sizeof(Conserved);
    if(bytes>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("MPI halo exceeds INT_MAX");
    const int count=static_cast<int>(bytes),prev=(c.rank-1+c.ranks)%c.ranks,
              next=(c.rank+1)%c.ranks;
    Conserved* data=q.data().data();
    MPI_Sendrecv(data+q.k_begin()*plane,count,MPI_BYTE,prev,310,
                 data+q.k_end()*plane,count,MPI_BYTE,next,310,c.comm,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(data+(q.k_end()-q.ng())*plane,count,MPI_BYTE,next,311,
                 data,count,MPI_BYTE,prev,311,c.comm,MPI_STATUS_IGNORE);
}
} // namespace

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        const Options o=parse_args(argc,argv);
        if(o.n%ranks)throw std::runtime_error("resolution must divide MPI ranks");
        const int local_nz=o.n/ranks;if(local_nz<2)throw std::runtime_error("z slab too small");
        phys::gamma=blast3d::gamma;
        const double dz=(blast3d::x_max-blast3d::x_min)/o.n;
        const double z0=blast3d::x_min+rank*local_nz*dz,z1=z0+local_nz*dz;
        Grid3D old(o.n,o.n,local_nz,2,blast3d::x_min,blast3d::x_max,
                   blast3d::x_min,blast3d::x_max,z0,z1);
        for(int k=0;k<old.total_nz();++k)for(int j=0;j<old.total_ny();++j)
            for(int i=0;i<old.total_nx();++i)old(i,j,k)=blast3d::initial_state(
                old.x_center(i),old.y_center(j),old.z_center(k));
        const auto bc=blast3d::boundary_conditions();apply_boundary_x(old,bc);
        apply_boundary_y(old,bc);Context ctx{MPI_COMM_WORLD,rank,ranks};
        exchange_z(old,&ctx);
        Grid3D ux=old,uy=old,next=old;CpuWorkspace3D ws;ws.init(o.n,o.n,local_nz);
        double t=0;int step=0;MPI_Barrier(MPI_COMM_WORLD);
        const auto start=std::chrono::steady_clock::now();
        while(t<o.t_end-1e-14) {
            double local=compute_dt_cpu(old,o.cfl),dt=0;
            MPI_Allreduce(&local,&dt,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
            dt=std::min(dt,o.t_end-t);
            advance_cpu_distributed_z(old,ux,uy,next,dt,ws,o.solver,bc,
                                      exchange_z,&ctx);
            std::swap(old,next);t+=dt;++step;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count(),mcells=
            static_cast<double>(step)*o.n*o.n*o.n/seconds/1e6;
        if(rank==0)std::cout<<"[MPI3D] nx="<<o.n<<" ny="<<o.n<<" nz="<<o.n
                            <<" ranks="<<ranks<<" steps="<<step
                            <<" elapsed_s="<<seconds
                            <<" Mcell_updates_s="<<mcells<<"\n";
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        if(rank==0)std::cerr<<"Error: "<<e.what()<<"\n";
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
