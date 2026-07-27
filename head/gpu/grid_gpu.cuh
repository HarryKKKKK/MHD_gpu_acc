#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include "types.hpp"

// Device structure-of-arrays storage for the five Euler conserved fields.
class Grid2DGPU {
public:
    Grid2DGPU()=default;
    Grid2DGPU(int nx,int ny,int ng,double x0,double x1,double y0,double y1) {
        allocate(nx,ny,ng,x0,x1,y0,y1);
    }
    Grid2DGPU(const Grid2DGPU&)=delete;
    Grid2DGPU& operator=(const Grid2DGPU&)=delete;
    Grid2DGPU(Grid2DGPU&& o) noexcept { move_from(std::move(o)); }
    Grid2DGPU& operator=(Grid2DGPU&& o) noexcept {
        if(this!=&o){release();move_from(std::move(o));} return *this;
    }
    ~Grid2DGPU(){release();}

    void allocate(int nx,int ny,int ng,double x0,double x1,double y0,double y1) {
        release(); nx_=nx;ny_=ny;ng_=ng;x0_=x0;x1_=x1;y0_=y0;y1_=y1;
        dx_=(x1-x0)/nx;dy_=(y1-y0)/ny;
        const std::size_t bytes=num_cells()*sizeof(double);
        check(cudaMalloc(&rho_,bytes)); check(cudaMalloc(&rhou_,bytes));
        check(cudaMalloc(&rhov_,bytes));check(cudaMalloc(&rhow_,bytes));
        check(cudaMalloc(&E_,bytes));
    }
    void release() {
        if(rho_)cudaFree(rho_);if(rhou_)cudaFree(rhou_);
        if(rhov_)cudaFree(rhov_);if(rhow_)cudaFree(rhow_);if(E_)cudaFree(E_);
        rho_=rhou_=rhov_=rhow_=E_=nullptr;nx_=ny_=ng_=0;
    }
    int nx()const{return nx_;} int ny()const{return ny_;} int ng()const{return ng_;}
    int total_nx()const{return nx_+2*ng_;} int total_ny()const{return ny_+2*ng_;}
    int i_begin()const{return ng_;}int i_end()const{return ng_+nx_;}
    int j_begin()const{return ng_;}int j_end()const{return ng_+ny_;}
    double x_min()const{return x0_;}double x_max()const{return x1_;}
    double y_min()const{return y0_;}double y_max()const{return y1_;}
    double dx()const{return dx_;}double dy()const{return dy_;}
    std::size_t num_cells()const {
        return static_cast<std::size_t>(total_nx())*total_ny();
    }
    double* rho_ptr(){return rho_;} double* rhou_ptr(){return rhou_;}
    double* rhov_ptr(){return rhov_;}double* rhow_ptr(){return rhow_;}
    double* E_ptr(){return E_;}
    const double* rho_ptr()const{return rho_;}
    const double* rhou_ptr()const{return rhou_;}
    const double* rhov_ptr()const{return rhov_;}
    const double* rhow_ptr()const{return rhow_;}
    const double* E_ptr()const{return E_;}

    void upload_from_aos(const std::vector<Conserved>& h) {
        if(h.size()!=num_cells())throw std::runtime_error("GPU upload size mismatch");
        const std::size_t n=num_cells(),bytes=n*sizeof(double);
        std::vector<double> rho(n),rhou(n),rhov(n),rhow(n),E(n);
        for(std::size_t i=0;i<n;++i) {
            rho[i]=h[i].rho;rhou[i]=h[i].rhou;rhov[i]=h[i].rhov;
            rhow[i]=h[i].rhow;E[i]=h[i].E;
        }
        check(cudaMemcpy(rho_,rho.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(rhou_,rhou.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(rhov_,rhov.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(rhow_,rhow.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(E_,E.data(),bytes,cudaMemcpyHostToDevice));
    }
    void download_to_aos(std::vector<Conserved>& h)const {
        const std::size_t n=num_cells(),bytes=n*sizeof(double);
        h.resize(n);std::vector<double> rho(n),rhou(n),rhov(n),rhow(n),E(n);
        check(cudaMemcpy(rho.data(),rho_,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(rhou.data(),rhou_,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(rhov.data(),rhov_,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(rhow.data(),rhow_,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(E.data(),E_,bytes,cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<n;++i)h[i]={rho[i],rhou[i],rhov[i],rhow[i],E[i]};
    }
    void swap(Grid2DGPU& o) {
        std::swap(rho_,o.rho_);std::swap(rhou_,o.rhou_);
        std::swap(rhov_,o.rhov_);std::swap(rhow_,o.rhow_);std::swap(E_,o.E_);
    }
private:
    int nx_=0,ny_=0,ng_=0;double x0_=0,x1_=1,y0_=0,y1_=1,dx_=0,dy_=0;
    double *rho_=nullptr,*rhou_=nullptr,*rhov_=nullptr,*rhow_=nullptr,*E_=nullptr;
    static void check(cudaError_t e) {
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    void move_from(Grid2DGPU&& o) {
        nx_=o.nx_;ny_=o.ny_;ng_=o.ng_;x0_=o.x0_;x1_=o.x1_;
        y0_=o.y0_;y1_=o.y1_;dx_=o.dx_;dy_=o.dy_;
        rho_=o.rho_;rhou_=o.rhou_;rhov_=o.rhov_;rhow_=o.rhow_;E_=o.E_;
        o.rho_=o.rhou_=o.rhov_=o.rhow_=o.E_=nullptr;o.nx_=o.ny_=o.ng_=0;
    }
};

struct Grid2DGPUView {
    int nx,ny,ng;double x_min,x_max,y_min,y_max,dx,dy;
    double *rho,*rhou,*rhov,*rhow,*E;
    __host__ __device__ int total_nx()const{return nx+2*ng;}
    __host__ __device__ int total_ny()const{return ny+2*ng;}
    __host__ __device__ int i_begin()const{return ng;}
    __host__ __device__ int i_end()const{return ng+nx;}
    __host__ __device__ int j_begin()const{return ng;}
    __host__ __device__ int j_end()const{return ng+ny;}
    __host__ __device__ int flat_index(int i,int j)const{return j*total_nx()+i;}
};
struct ConstGrid2DGPUView {
    int nx,ny,ng;double x_min,x_max,y_min,y_max,dx,dy;
    const double *rho,*rhou,*rhov,*rhow,*E;
    __host__ __device__ int total_nx()const{return nx+2*ng;}
    __host__ __device__ int total_ny()const{return ny+2*ng;}
    __host__ __device__ int i_begin()const{return ng;}
    __host__ __device__ int i_end()const{return ng+nx;}
    __host__ __device__ int j_begin()const{return ng;}
    __host__ __device__ int j_end()const{return ng+ny;}
    __host__ __device__ int flat_index(int i,int j)const{return j*total_nx()+i;}
};
inline Grid2DGPUView make_view(Grid2DGPU& g) {
    return {g.nx(),g.ny(),g.ng(),g.x_min(),g.x_max(),g.y_min(),g.y_max(),
            g.dx(),g.dy(),g.rho_ptr(),g.rhou_ptr(),g.rhov_ptr(),g.rhow_ptr(),
            g.E_ptr()};
}
inline ConstGrid2DGPUView make_view(const Grid2DGPU& g) {
    return {g.nx(),g.ny(),g.ng(),g.x_min(),g.x_max(),g.y_min(),g.y_max(),
            g.dx(),g.dy(),g.rho_ptr(),g.rhou_ptr(),g.rhov_ptr(),g.rhow_ptr(),
            g.E_ptr()};
}
