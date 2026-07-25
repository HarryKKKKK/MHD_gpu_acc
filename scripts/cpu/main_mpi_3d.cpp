// Pure-MPI 3D GLM-MHD benchmark driver.
//
// The global cube is decomposed into contiguous z slabs. Each rank owns the
// full x/y cross-section and exchanges two z ghost planes with its periodic
// neighbours between directional sweeps.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "blast3d_case.hpp"
#include "blast3d_extreme_case.hpp"
#include "cpu/boundary3d_cpu.hpp"
#include "cpu/grid3d_cpu.hpp"
#include "cpu/solver3d_cpu.hpp"
#include "imtg3d_case.hpp"

namespace {

enum class Case3D { Blast, BlastExtreme, IMTG };

struct Options {
    int n=-1;
    int snapshots=-1;
    double t_end=-1.0;
    double cfl=-1.0;
    RiemannSolver solver=RiemannSolver::HLLD;
    Case3D test_case=Case3D::Blast;
};

Options parse_args(int argc,char** argv) {
    Options o;
    for(int a=1;a<argc;++a) {
        const std::string s=argv[a];
        auto value=[&]() -> std::string {
            if(a+1>=argc) throw std::runtime_error("missing value after "+s);
            return argv[++a];
        };
        if(s=="--case") {
            const auto v=value();
            if(v=="blast" || v=="blast_athena") o.test_case=Case3D::Blast;
            else if(v=="blast_extreme") o.test_case=Case3D::BlastExtreme;
            else if(v=="imtg") o.test_case=Case3D::IMTG;
            else throw std::runtime_error("unknown case: "+v);
        } else if(s=="--resolution") {
            o.n=std::stoi(value());
        } else if(s=="--snapshots") {
            o.snapshots=std::stoi(value());
        } else if(s=="--t-end") {
            o.t_end=std::stod(value());
        } else if(s=="--cfl") {
            o.cfl=std::stod(value());
        } else if(s=="--solver") {
            const auto v=value();
            if(v=="hll") o.solver=RiemannSolver::HLL;
            else if(v=="hllc") o.solver=RiemannSolver::HLLC;
            else if(v=="hlld") o.solver=RiemannSolver::HLLD;
            else if(v=="force") o.solver=RiemannSolver::FORCE;
            else throw std::runtime_error("unknown solver: "+v);
        } else if(s=="--no-out" || s=="--timing-only") {
            // This benchmark-only MPI driver never gathers volume output.
        } else {
            throw std::runtime_error("unknown argument: "+s);
        }
    }
    const bool imtg=o.test_case==Case3D::IMTG;
    const bool extreme=o.test_case==Case3D::BlastExtreme;
    if(o.n<0) o.n=128;
    if(o.snapshots<0) o.snapshots=imtg?12:5;
    if(o.t_end<0) o.t_end=imtg?imtg3d::t_end:
        (extreme?blast3d_extreme::t_end:blast3d::t_end);
    if(o.cfl<0) o.cfl=imtg?imtg3d::recommended_cfl:
        (extreme?blast3d_extreme::recommended_cfl:blast3d::recommended_cfl);
    if(o.n<8 || o.snapshots<1 || o.t_end<=0.0 || o.cfl<=0.0)
        throw std::runtime_error(
            "require resolution>=8, snapshots>=1, t_end>0, cfl>0");
    return o;
}

struct ZExchangeContext {
    MPI_Comm comm=MPI_COMM_WORLD;
    int rank=0;
    int ranks=1;
};

void exchange_z_halos(Grid3D& q,void* opaque) {
    auto& context=*static_cast<ZExchangeContext*>(opaque);
    if(context.ranks==1) {
        BoundaryConfig3D periodic;
        periodic.z_min=periodic.z_max=BoundaryType::Periodic;
        apply_boundary_z(q,periodic);
        return;
    }

    static_assert(std::is_trivially_copyable<Conserved>::value,
                  "Conserved must be safe for byte-wise MPI transfer");
    if(q.ng()!=2 || q.nz()<2)
        throw std::runtime_error("MPI z slabs require ng=2 and local nz>=2");

    const std::size_t plane=
        static_cast<std::size_t>(q.total_nx())*q.total_ny();
    const std::size_t exchange_bytes=
        static_cast<std::size_t>(q.ng())*plane*sizeof(Conserved);
    if(exchange_bytes>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("MPI halo message exceeds INT_MAX bytes");
    const int count=static_cast<int>(exchange_bytes);
    const int previous=(context.rank-1+context.ranks)%context.ranks;
    const int next=(context.rank+1)%context.ranks;
    Conserved* const data=q.data().data();

    // First two active planes go to the previous rank and become the next
    // rank's upper ghosts. Receive our upper ghosts at k_end.
    MPI_Sendrecv(
        data+static_cast<std::size_t>(q.k_begin())*plane,count,MPI_BYTE,
        previous,310,
        data+static_cast<std::size_t>(q.k_end())*plane,count,MPI_BYTE,
        next,310,context.comm,MPI_STATUS_IGNORE);

    // Last two active planes go to the next rank. The previous rank's final
    // planes are received directly into our two lower ghost planes.
    MPI_Sendrecv(
        data+static_cast<std::size_t>(q.k_end()-q.ng())*plane,count,MPI_BYTE,
        next,311,
        data,count,MPI_BYTE,
        previous,311,context.comm,MPI_STATUS_IGNORE);
}

const char* solver_name(RiemannSolver solver) {
    if(solver==RiemannSolver::HLL) return "HLL";
    if(solver==RiemannSolver::HLLC) return "HLLC";
    if(solver==RiemannSolver::HLLD) return "HLLD";
    return "FORCE";
}

} // namespace

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);

    try {
        const Options o=parse_args(argc,argv);
        if(o.n%ranks!=0)
            throw std::runtime_error(
                "global resolution must be divisible by the MPI rank count");
        const int local_nz=o.n/ranks;
        if(local_nz<2)
            throw std::runtime_error(
                "each MPI z slab must contain at least two active planes");

        const bool imtg=o.test_case==Case3D::IMTG;
        const bool extreme=o.test_case==Case3D::BlastExtreme;
        const double lo=imtg?imtg3d::x_min:
            (extreme?blast3d_extreme::x_min:blast3d::x_min);
        const double hi=imtg?imtg3d::x_max:
            (extreme?blast3d_extreme::x_max:blast3d::x_max);
        phys::gamma=imtg?imtg3d::gamma:
            (extreme?blast3d_extreme::gamma:blast3d::gamma);

        const double dz=(hi-lo)/o.n;
        const double local_z_min=lo+rank*local_nz*dz;
        const double local_z_max=local_z_min+local_nz*dz;
        Grid3D old(o.n,o.n,local_nz,2,
                   lo,hi,lo,hi,local_z_min,local_z_max);
        for(int k=0;k<old.total_nz();++k)
            for(int j=0;j<old.total_ny();++j)
                for(int i=0;i<old.total_nx();++i)
                    old(i,j,k)=imtg
                        ? imtg3d::initial_state(
                            old.x_center(i),old.y_center(j),old.z_center(k))
                        : (extreme
                            ? blast3d_extreme::initial_state(
                                old.x_center(i),old.y_center(j),old.z_center(k))
                            : blast3d::initial_state(
                                old.x_center(i),old.y_center(j),old.z_center(k)));

        const BoundaryConfig3D bc=imtg
            ? imtg3d::boundary_conditions()
            : (extreme
                ? blast3d_extreme::boundary_conditions()
                : blast3d::boundary_conditions());
        apply_boundary_x(old,bc);
        apply_boundary_y(old,bc);
        ZExchangeContext exchange{MPI_COMM_WORLD,rank,ranks};
        exchange_z_halos(old,&exchange);

        Grid3D ux=old,uy=old,next=old;
        CpuWorkspace3D workspace;
        workspace.init(o.n,o.n,local_nz);

        std::vector<double> targets;
        for(int snapshot=0;snapshot<=o.snapshots;++snapshot)
            targets.push_back(
                o.t_end*static_cast<double>(snapshot)/o.snapshots);

        if(rank==0) {
            const char* case_name=imtg?"imtg":
                (extreme?"blast_extreme":"blast");
            std::cout<<"=== Pure MPI 3D GLM-MHD benchmark ===\n"
                     <<"  case       : "<<case_name<<"\n"
                     <<"  grid       : "<<o.n<<" x "<<o.n<<" x "<<o.n<<"\n"
                     <<"  ranks      : "<<ranks<<"\n"
                     <<"  z/rank     : "<<local_nz<<"\n"
                     <<"  solver     : "<<solver_name(o.solver)<<"\n"
                     <<"  t_end/CFL  : "<<o.t_end<<" / "<<o.cfl<<"\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double start=MPI_Wtime();
        double time=0.0;
        int step=0,frame=1;
        while(time<o.t_end-1.0e-14) {
            const double local_raw=compute_dt_cpu(old,o.cfl);
            double global_raw=0.0;
            MPI_Allreduce(
                &local_raw,&global_raw,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
            if(!std::isfinite(global_raw) || global_raw<=0.0)
                throw std::runtime_error("invalid global MPI timestep");
            phys::ch_glm=o.cfl*std::min({old.dx(),old.dy(),old.dz()})/
                global_raw;
            const double dt=std::min(
                global_raw,targets[static_cast<std::size_t>(frame)]-time);
            advance_cpu_distributed_z(
                old,ux,uy,next,dt,workspace,o.solver,bc,
                exchange_z_halos,&exchange);
            std::swap(old,next);
            time+=dt;
            ++step;
            if(time>=targets[static_cast<std::size_t>(frame)]-1.0e-12)
                ++frame;
            if(rank==0 && (step%20==0 || time>=o.t_end-1.0e-14))
                std::cout<<"  step "<<step<<" t="<<time<<" dt="<<dt<<"\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double local_elapsed=MPI_Wtime()-start;
        double elapsed=0.0;
        MPI_Reduce(
            &local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(rank==0) {
            const double mcells=
                static_cast<double>(step)*o.n*o.n*o.n/elapsed/1.0e6;
            std::cout<<"[MPI3D] nx="<<o.n<<" ny="<<o.n<<" nz="<<o.n
                     <<" ranks="<<ranks<<" steps="<<step
                     <<" elapsed_s="<<elapsed
                     <<" Mcell_updates_s="<<mcells<<"\n";
        }
        MPI_Finalize();
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"[rank "<<rank<<"] Error: "<<error.what()<<"\n";
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
