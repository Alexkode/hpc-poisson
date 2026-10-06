CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CUDA_ARCH ?= 80

.PHONY: all clean mpi openmp cuda
all: build/poisson
build:
	mkdir -p build
build/poisson: src/poisson.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@
openmp: | build
	$(CXX) $(CXXFLAGS) -fopenmp src/poisson.cpp -o build/poisson-omp
mpi: | build
	mpicxx $(CXXFLAGS) -DUSE_MPI -fopenmp src/poisson.cpp -o build/poisson-mpi
	mpicxx $(CXXFLAGS) -DUSE_MPI -DFIELD_FLOAT -fopenmp src/poisson.cpp -o build/poisson-mixed
cuda: | build
	nvcc -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) src/poisson_cuda.cu -o build/poisson-cuda
clean:
	rm -rf build
