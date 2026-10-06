#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void checked(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Grid { int n; size_t plane,total; double a,diag; };

__global__ void jacobi(const double* u,double* v,const double* f,Grid g,double omega) {
    size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    size_t count=size_t(g.n)*g.n*g.n;
    if(q>=count) return;
    int x=int(q%g.n)+1, y=int((q/g.n)%g.n)+1, z=int(q/(size_t(g.n)*g.n))+1;
    size_t i=size_t(z)*g.plane+size_t(y)*(g.n+2)+x;
    double candidate=(f[i]+g.a*(u[i-1]+u[i+1]+u[i-g.n-2]+u[i+g.n+2]+
                                            u[i-g.plane]+u[i+g.plane]))/g.diag;
    v[i]=(1-omega)*u[i]+omega*candidate;
}

__global__ void norms(const double* u,const double* f,const double* exact,double* sums,Grid g) {
    __shared__ double s[4][256];
    size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    int t=threadIdx.x;
    for(int k=0;k<4;++k) s[k][t]=0;
    if(q<size_t(g.n)*g.n*g.n) {
        int x=int(q%g.n)+1, y=int((q/g.n)%g.n)+1,z=int(q/(size_t(g.n)*g.n))+1;
        size_t i=size_t(z)*g.plane+size_t(y)*(g.n+2)+x;
        double r=f[i]-g.diag*u[i]+g.a*(u[i-1]+u[i+1]+u[i-g.n-2]+u[i+g.n+2]+
                                                        u[i-g.plane]+u[i+g.plane]);
        double e=u[i]-exact[i];
        s[0][t]=r*r; s[1][t]=f[i]*f[i]; s[2][t]=e*e; s[3][t]=exact[i]*exact[i];
    }
    __syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2) {
        if(t<stride) for(int k=0;k<4;++k) s[k][t]+=s[k][t+stride];
        __syncthreads();
    }
    if(t==0) for(int k=0;k<4;++k) atomicAdd(sums+k,s[k][0]);
}

int main(int argc,char** argv) {
    try {
        int n=24,iterations=10000,check=50; double tolerance=1e-7,omega=.8;
        std::string output;
        for(int i=1;i<argc;++i) {
            std::string k=argv[i];
            if(k=="--help") {std::cout<<"poisson-cuda --n N --iterations N --tolerance T --check N --omega W --output FILE\n";return 0;}
            if(++i>=argc) throw std::invalid_argument("Missing value");
            std::string v=argv[i];
            if(k=="--n")n=std::stoi(v);
            else if(k=="--iterations")iterations=std::stoi(v);
            else if(k=="--check")check=std::stoi(v);
            else if(k=="--tolerance")tolerance=std::stod(v);
            else if(k=="--omega")omega=std::stod(v);
            else if(k=="--output")output=v;
            else throw std::invalid_argument("Unknown argument "+k);
        }
        if(n<1||iterations<1||check<1||!std::isfinite(tolerance)||tolerance<0||
           !std::isfinite(omega)||omega<=0||omega>1) throw std::invalid_argument("Invalid configuration");
        Grid g{n,size_t(n+2)*(n+2),size_t(n+2)*(n+2)*(n+2),std::pow(n+1.,2),6*std::pow(n+1.,2)};
        std::vector<double> f(g.total),exact(g.total);
        const double pi=std::acos(-1.);
        for(int z=1;z<=n;++z)for(int y=1;y<=n;++y)for(int x=1;x<=n;++x) {
            size_t i=size_t(z)*g.plane+size_t(y)*(n+2)+x;
            exact[i]=std::sin(pi*x/(n+1))*std::sin(pi*y/(n+1))*std::sin(pi*z/(n+1));
            f[i]=3*pi*pi*exact[i];
        }
        double *u,*v,*df,*de,*ds; size_t bytes=g.total*sizeof(double);
        checked(cudaMalloc(&u,bytes));checked(cudaMalloc(&v,bytes));checked(cudaMalloc(&df,bytes));
        checked(cudaMalloc(&de,bytes));checked(cudaMalloc(&ds,4*sizeof(double)));
        checked(cudaMemset(u,0,bytes));checked(cudaMemset(v,0,bytes));
        checked(cudaMemcpy(df,f.data(),bytes,cudaMemcpyHostToDevice));
        checked(cudaMemcpy(de,exact.data(),bytes,cudaMemcpyHostToDevice));
        int blocks=int((size_t(n)*n*n+255)/256);
        auto measure=[&]() {
            checked(cudaMemset(ds,0,4*sizeof(double)));
            norms<<<blocks,256>>>(u,df,de,ds,g);checked(cudaGetLastError());
            std::vector<double> s(4);checked(cudaMemcpy(s.data(),ds,4*sizeof(double),cudaMemcpyDeviceToHost));
            return std::vector<double>{std::sqrt(s[0]/s[1]),std::sqrt(s[2]/s[3])};
        };
        jacobi<<<blocks,256>>>(u,v,df,g,omega);checked(cudaGetLastError());
        checked(cudaDeviceSynchronize());checked(cudaMemset(v,0,bytes));
        cudaEvent_t start,stop;checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
        checked(cudaEventRecord(start));int completed=0;
        for(int it=1;it<=iterations;++it) {
            jacobi<<<blocks,256>>>(u,v,df,g,omega);checked(cudaGetLastError());std::swap(u,v);completed=it;
            if(tolerance>0 && it%check==0 && measure()[0]<=tolerance)break;
        }
        checked(cudaEventRecord(stop));checked(cudaEventSynchronize(stop));
        float ms;checked(cudaEventElapsedTime(&ms,start,stop));auto m=measure();
        cudaDeviceProp prop;checked(cudaGetDeviceProperties(&prop,0));
        std::ostringstream o;
        o<<std::setprecision(12)<<"{\"backend\":\"cuda\",\"precision\":\"fp64\",\"device\":\""<<prop.name
         <<"\",\"nx\":"<<n<<",\"ny\":"<<n<<",\"nz\":"<<n<<",\"ranks\":1,\"threads\":1,\"iterations\":"
         <<completed<<",\"seconds\":"<<ms/1000.<<",\"relative_residual\":"<<m[0]
         <<",\"relative_solution_error\":"<<m[1]<<",\"updates_per_second\":"<<double(n)*n*n*completed/(ms/1000.)<<"}";
        std::cout<<o.str()<<'\n';
        if(!output.empty()){std::ofstream file(output);if(!file)throw std::runtime_error("Cannot open output");file<<o.str()<<'\n';}
        for(auto p:{u,v,df,de,ds})checked(cudaFree(p));
        checked(cudaEventDestroy(start));checked(cudaEventDestroy(stop));return 0;
    }catch(const std::exception& e){std::cerr<<"poisson-cuda: "<<e.what()<<'\n';return 1;}
}
