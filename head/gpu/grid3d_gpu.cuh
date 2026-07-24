#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "types.hpp"

inline void cuda3d_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
}

struct Grid3DGPUView {
    int nx, ny, nz, ng;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    double dx, dy, dz;
    Conserved* cells;

    __host__ __device__ int total_nx() const { return nx + 2*ng; }
    __host__ __device__ int total_ny() const { return ny + 2*ng; }
    __host__ __device__ int total_nz() const { return nz + 2*ng; }
    __host__ __device__ int i_begin() const { return ng; }
    __host__ __device__ int i_end() const { return ng+nx; }
    __host__ __device__ int j_begin() const { return ng; }
    __host__ __device__ int j_end() const { return ng+ny; }
    __host__ __device__ int k_begin() const { return ng; }
    __host__ __device__ int k_end() const { return ng+nz; }
    __host__ __device__ std::size_t flat_index(int i,int j,int k) const {
        return (static_cast<std::size_t>(k)*total_ny()+j)*total_nx()+i;
    }
};

struct ConstGrid3DGPUView {
    int nx, ny, nz, ng;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    double dx, dy, dz;
    const Conserved* cells;

    __host__ __device__ int total_nx() const { return nx + 2*ng; }
    __host__ __device__ int total_ny() const { return ny + 2*ng; }
    __host__ __device__ int total_nz() const { return nz + 2*ng; }
    __host__ __device__ int i_begin() const { return ng; }
    __host__ __device__ int i_end() const { return ng+nx; }
    __host__ __device__ int j_begin() const { return ng; }
    __host__ __device__ int j_end() const { return ng+ny; }
    __host__ __device__ int k_begin() const { return ng; }
    __host__ __device__ int k_end() const { return ng+nz; }
    __host__ __device__ std::size_t flat_index(int i,int j,int k) const {
        return (static_cast<std::size_t>(k)*total_ny()+j)*total_nx()+i;
    }
};

class Grid3DGPU {
public:
    Grid3DGPU() = default;
    Grid3DGPU(int nx,int ny,int nz,int ng,
              double x0,double x1,double y0,double y1,double z0,double z1) {
        allocate(nx,ny,nz,ng,x0,x1,y0,y1,z0,z1);
    }
    Grid3DGPU(const Grid3DGPU&)=delete;
    Grid3DGPU& operator=(const Grid3DGPU&)=delete;
    Grid3DGPU(Grid3DGPU&& o) noexcept { move_from(std::move(o)); }
    Grid3DGPU& operator=(Grid3DGPU&& o) noexcept {
        if(this!=&o){ release(); move_from(std::move(o)); }
        return *this;
    }
    ~Grid3DGPU(){ release(); }

    void allocate(int nx,int ny,int nz,int ng,
                  double x0,double x1,double y0,double y1,double z0,double z1) {
        release();
        nx_=nx; ny_=ny; nz_=nz; ng_=ng;
        x0_=x0; x1_=x1; y0_=y0; y1_=y1; z0_=z0; z1_=z1;
        dx_=(x1-x0)/nx; dy_=(y1-y0)/ny; dz_=(z1-z0)/nz;
        cuda3d_check(cudaMalloc(&cells_,num_cells()*sizeof(Conserved)),
                     "cudaMalloc Grid3DGPU");
    }
    void release() {
        if(cells_) cudaFree(cells_);
        cells_=nullptr; nx_=ny_=nz_=ng_=0;
    }
    int nx()const{return nx_;} int ny()const{return ny_;} int nz()const{return nz_;}
    int ng()const{return ng_;}
    int total_nx()const{return nx_+2*ng_;} int total_ny()const{return ny_+2*ng_;}
    int total_nz()const{return nz_+2*ng_;}
    int i_begin()const{return ng_;} int i_end()const{return ng_+nx_;}
    int j_begin()const{return ng_;} int j_end()const{return ng_+ny_;}
    int k_begin()const{return ng_;} int k_end()const{return ng_+nz_;}
    double x_min()const{return x0_;} double x_max()const{return x1_;}
    double y_min()const{return y0_;} double y_max()const{return y1_;}
    double z_min()const{return z0_;} double z_max()const{return z1_;}
    double dx()const{return dx_;} double dy()const{return dy_;} double dz()const{return dz_;}
    std::size_t num_cells()const {
        return static_cast<std::size_t>(total_nx())*total_ny()*total_nz();
    }
    Conserved* data(){return cells_;} const Conserved* data()const{return cells_;}

    void upload_from_aos(const std::vector<Conserved>& host) {
        if(host.size()!=num_cells())
            throw std::runtime_error("Grid3DGPU upload size mismatch");
        cuda3d_check(cudaMemcpy(cells_,host.data(),num_cells()*sizeof(Conserved),
                               cudaMemcpyHostToDevice),"Grid3DGPU upload");
    }
    void download_to_aos(std::vector<Conserved>& host)const {
        host.resize(num_cells());
        cuda3d_check(cudaMemcpy(host.data(),cells_,num_cells()*sizeof(Conserved),
                               cudaMemcpyDeviceToHost),"Grid3DGPU download");
    }
    void swap(Grid3DGPU& o){ std::swap(cells_,o.cells_); }

private:
    int nx_=0,ny_=0,nz_=0,ng_=0;
    double x0_=0,x1_=1,y0_=0,y1_=1,z0_=0,z1_=1,dx_=0,dy_=0,dz_=0;
    Conserved* cells_=nullptr;
    void move_from(Grid3DGPU&& o) {
        nx_=o.nx_;ny_=o.ny_;nz_=o.nz_;ng_=o.ng_;
        x0_=o.x0_;x1_=o.x1_;y0_=o.y0_;y1_=o.y1_;z0_=o.z0_;z1_=o.z1_;
        dx_=o.dx_;dy_=o.dy_;dz_=o.dz_;cells_=o.cells_;
        o.cells_=nullptr;o.nx_=o.ny_=o.nz_=o.ng_=0;
    }
};

inline Grid3DGPUView make_view(Grid3DGPU& g) {
    return {g.nx(),g.ny(),g.nz(),g.ng(),
            g.x_min(),g.x_max(),g.y_min(),g.y_max(),g.z_min(),g.z_max(),
            g.dx(),g.dy(),g.dz(),g.data()};
}
inline ConstGrid3DGPUView make_view(const Grid3DGPU& g) {
    return {g.nx(),g.ny(),g.nz(),g.ng(),
            g.x_min(),g.x_max(),g.y_min(),g.y_max(),g.z_min(),g.z_max(),
            g.dx(),g.dy(),g.dz(),g.data()};
}

