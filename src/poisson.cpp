#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef FIELD_FLOAT
using Real = float;
#else
using Real = double;
#endif

struct Config {
    int nx=24, ny=24, nz=24, iterations=10000, check=50;
    double tolerance=1e-7, omega=0.8;
    bool overlap=false;
    std::string output;
};

Config parse(int argc, char** argv) {
    Config c;
    for (int i=1; i<argc; ++i) {
        std::string k=argv[i];
        if (k=="--overlap") { c.overlap=true; continue; }
        if (k=="--help") {
            std::cout << "poisson [--n N | --nx N --ny N --nz N] --iterations N "
                         "--tolerance T --check N --omega W --overlap --output FILE\n"
                         "Use tolerance 0 for a fixed number of timed sweeps.\n";
            std::exit(0);
        }
        if (++i>=argc) throw std::invalid_argument("Missing value for "+k);
        std::string v=argv[i];
        if (k=="--n") c.nx=c.ny=c.nz=std::stoi(v);
        else if (k=="--nx") c.nx=std::stoi(v);
        else if (k=="--ny") c.ny=std::stoi(v);
        else if (k=="--nz") c.nz=std::stoi(v);
        else if (k=="--iterations") c.iterations=std::stoi(v);
        else if (k=="--check") c.check=std::stoi(v);
        else if (k=="--tolerance") c.tolerance=std::stod(v);
        else if (k=="--omega") c.omega=std::stod(v);
        else if (k=="--output") c.output=v;
        else throw std::invalid_argument("Unknown argument "+k);
    }
    if (std::min({c.nx,c.ny,c.nz,c.iterations,c.check})<1 ||
        !std::isfinite(c.omega) || c.omega<=0 || c.omega>1 ||
        !std::isfinite(c.tolerance) || c.tolerance<0)
        throw std::invalid_argument("Invalid grid, iteration count, tolerance or omega");
    return c;
}

struct Domain {
    int nx,ny,nz,offset,rank,ranks;
    size_t plane;
    double ax,ay,az,diagonal;
    std::vector<Real> u,v,f,exact;
    Domain(const Config& c, int r, int p):nx(c.nx),ny(c.ny),rank(r),ranks(p) {
        if (p>c.nz) throw std::invalid_argument("Ranks must not exceed nz");
        int base=c.nz/p, extra=c.nz%p;
        nz=base+(r<extra);
        offset=r*base+std::min(r,extra);
        plane=size_t(nx+2)*(ny+2);
        ax=std::pow(c.nx+1.,2); ay=std::pow(c.ny+1.,2); az=std::pow(c.nz+1.,2);
        diagonal=2*(ax+ay+az);
        u.resize(plane*(nz+2)); v=u; f=u; exact=u;
        const double pi=std::acos(-1.);
        for (int z=1;z<=nz;++z) for (int y=1;y<=ny;++y) for (int x=1;x<=nx;++x) {
            size_t i=index(x,y,z);
            double e=std::sin(pi*x/(nx+1))*std::sin(pi*y/(ny+1))*
                     std::sin(pi*(offset+z)/(c.nz+1));
            exact[i]=Real(e); f[i]=Real(3*pi*pi*e);
        }
    }
    size_t index(int x,int y,int z) const {return size_t(z)*plane+size_t(y)*(nx+2)+x;}
    void halo() {
#ifdef USE_MPI
        const int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<ranks?rank+1:MPI_PROC_NULL;
        MPI_Datatype type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
        MPI_Sendrecv(u.data()+plane,int(plane),type,prev,10,
                     u.data()+(nz+1)*plane,int(plane),type,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(u.data()+nz*plane,int(plane),type,next,11,
                     u.data(),int(plane),type,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
#endif
    }
    void update(int first,int last,double omega) {
        // Every output cell has one writer. Ghosts and physical boundaries remain zero.
#pragma omp parallel for collapse(2) schedule(static)
        for (int z=first;z<=last;++z) for (int y=1;y<=ny;++y) for (int x=1;x<=nx;++x) {
            size_t i=index(x,y,z);
            double candidate=(double(f[i])+ax*(double(u[i-1])+double(u[i+1]))+
                ay*(double(u[i-nx-2])+double(u[i+nx+2]))+
                az*(double(u[i-plane])+double(u[i+plane])))/diagonal;
            v[i]=Real((1-omega)*double(u[i])+omega*candidate);
        }
    }
    void sweep(double omega,bool overlap) {
#ifdef USE_MPI
        if (overlap && ranks>1) {
            int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<ranks?rank+1:MPI_PROC_NULL;
            MPI_Datatype type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
            MPI_Request req[4];
            MPI_Irecv(u.data(),int(plane),type,prev,11,MPI_COMM_WORLD,&req[0]);
            MPI_Irecv(u.data()+(nz+1)*plane,int(plane),type,next,10,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(u.data()+plane,int(plane),type,prev,10,MPI_COMM_WORLD,&req[2]);
            MPI_Isend(u.data()+nz*plane,int(plane),type,next,11,MPI_COMM_WORLD,&req[3]);
            update(2,nz-1,omega);
            MPI_Waitall(4,req,MPI_STATUSES_IGNORE);
            update(1,1,omega);
            if (nz>1) update(nz,nz,omega);
        } else
#endif
        { halo(); update(1,nz,omega); }
        u.swap(v);
    }
    std::vector<double> metrics() {
        halo();
        double residual=0,source=0,error=0,norm=0;
#pragma omp parallel for collapse(2) reduction(+:residual,source,error,norm)
        for (int z=1;z<=nz;++z) for (int y=1;y<=ny;++y) for (int x=1;x<=nx;++x) {
            size_t i=index(x,y,z);
            double au=diagonal*double(u[i])-ax*(double(u[i-1])+double(u[i+1]))-
                ay*(double(u[i-nx-2])+double(u[i+nx+2]))-
                az*(double(u[i-plane])+double(u[i+plane]));
            double r=double(f[i])-au, e=double(u[i])-double(exact[i]);
            residual+=r*r; source+=double(f[i])*double(f[i]);
            error+=e*e; norm+=double(exact[i])*double(exact[i]);
        }
        double sums[4]={residual,source,error,norm};
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE,sums,4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
        return {std::sqrt(sums[0]/sums[1]),std::sqrt(sums[2]/sums[3])};
    }
};

int main(int argc,char** argv) {
    int rank=0,ranks=1;
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
    try {
        Config c=parse(argc,argv); Domain d(c,rank,ranks);
        // Warm-up a separate field to preserve the prescribed zero initial condition.
        d.sweep(c.omega,c.overlap);
        std::fill(d.u.begin(),d.u.end(),Real(0)); std::fill(d.v.begin(),d.v.end(),Real(0));
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto start=std::chrono::steady_clock::now();
        int completed=0; double residual=1;
        for (int it=1;it<=c.iterations;++it) {
            d.sweep(c.omega,c.overlap); completed=it;
            if (c.tolerance>0 && it%c.check==0) {
                residual=d.metrics()[0]; if (residual<=c.tolerance) break;
            }
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE,&seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
        auto m=d.metrics();
        int threads=1;
#ifdef _OPENMP
        threads=omp_get_max_threads();
#endif
        if (rank==0) {
            std::ostringstream o;
            o << std::setprecision(12) << "{\"backend\":\"cpu\",\"precision\":\""
              << (sizeof(Real)==4?"fp32-field-fp64-arithmetic":"fp64")
              << "\",\"nx\":" << c.nx << ",\"ny\":" << c.ny << ",\"nz\":" << c.nz
              << ",\"ranks\":" << ranks << ",\"threads\":" << threads
              << ",\"overlap\":" << (c.overlap?"true":"false") << ",\"iterations\":" << completed
              << ",\"seconds\":" << seconds << ",\"relative_residual\":" << m[0]
              << ",\"relative_solution_error\":" << m[1] << ",\"updates_per_second\":"
              << double(c.nx)*c.ny*c.nz*completed/seconds << "}";
            std::cout << o.str() << '\n';
            if (!c.output.empty()) {
                std::ofstream file(c.output); if (!file) throw std::runtime_error("Cannot open output");
                file << o.str() << '\n';
            }
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "poisson: " << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD,1);
#endif
        return 1;
    }
}
