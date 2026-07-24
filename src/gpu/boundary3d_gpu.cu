#include "gpu/boundary3d_gpu.cuh"

namespace {

__device__ int source_index(bool lower,int g,int begin,int end,BoundaryType type) {
    if(lower) return type==BoundaryType::Periodic ? end-1-g : begin;
    return type==BoundaryType::Periodic ? begin+g : end-1;
}

__global__ void boundary_x_kernel(Grid3DGPUView q,BoundaryType lo,BoundaryType hi) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    const int count=q.total_ny()*q.total_nz();
    if(p>=count) return;
    const int j=p%q.total_ny(), k=p/q.total_ny();
    for(int g=0;g<q.ng;++g) {
        q.cells[q.flat_index(q.i_begin()-1-g,j,k)]=
            q.cells[q.flat_index(source_index(true,g,q.i_begin(),q.i_end(),lo),j,k)];
        q.cells[q.flat_index(q.i_end()+g,j,k)]=
            q.cells[q.flat_index(source_index(false,g,q.i_begin(),q.i_end(),hi),j,k)];
    }
}

__global__ void boundary_y_kernel(Grid3DGPUView q,BoundaryType lo,BoundaryType hi) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    const int count=q.total_nx()*q.total_nz();
    if(p>=count) return;
    const int i=p%q.total_nx(), k=p/q.total_nx();
    for(int g=0;g<q.ng;++g) {
        q.cells[q.flat_index(i,q.j_begin()-1-g,k)]=
            q.cells[q.flat_index(i,source_index(true,g,q.j_begin(),q.j_end(),lo),k)];
        q.cells[q.flat_index(i,q.j_end()+g,k)]=
            q.cells[q.flat_index(i,source_index(false,g,q.j_begin(),q.j_end(),hi),k)];
    }
}

__global__ void boundary_z_kernel(Grid3DGPUView q,BoundaryType lo,BoundaryType hi) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    const int count=q.total_nx()*q.total_ny();
    if(p>=count) return;
    const int i=p%q.total_nx(), j=p/q.total_nx();
    for(int g=0;g<q.ng;++g) {
        q.cells[q.flat_index(i,j,q.k_begin()-1-g)]=
            q.cells[q.flat_index(i,j,source_index(true,g,q.k_begin(),q.k_end(),lo))];
        q.cells[q.flat_index(i,j,q.k_end()+g)]=
            q.cells[q.flat_index(i,j,source_index(false,g,q.k_begin(),q.k_end(),hi))];
    }
}

void check_launch(const char* name) {
    cuda3d_check(cudaGetLastError(),name);
}

} // namespace

void apply_boundary_x_gpu(Grid3DGPU& q,const BoundaryConfig3D& bc) {
    constexpr int threads=256;
    const int n=q.total_ny()*q.total_nz();
    boundary_x_kernel<<<(n+threads-1)/threads,threads>>>(
        make_view(q),bc.x_min,bc.x_max);
    check_launch("boundary_x_kernel");
}
void apply_boundary_y_gpu(Grid3DGPU& q,const BoundaryConfig3D& bc) {
    constexpr int threads=256;
    const int n=q.total_nx()*q.total_nz();
    boundary_y_kernel<<<(n+threads-1)/threads,threads>>>(
        make_view(q),bc.y_min,bc.y_max);
    check_launch("boundary_y_kernel");
}
void apply_boundary_z_gpu(Grid3DGPU& q,const BoundaryConfig3D& bc) {
    constexpr int threads=256;
    const int n=q.total_nx()*q.total_ny();
    boundary_z_kernel<<<(n+threads-1)/threads,threads>>>(
        make_view(q),bc.z_min,bc.z_max);
    check_launch("boundary_z_kernel");
}
void apply_boundary_gpu(Grid3DGPU& q,const BoundaryConfig3D& bc) {
    apply_boundary_x_gpu(q,bc);
    apply_boundary_y_gpu(q,bc);
    apply_boundary_z_gpu(q,bc);
}

