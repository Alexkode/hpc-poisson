#!/bin/bash
#SBATCH --job-name=poisson-precision
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load cesga/2025 gcc/13.4.0
make openmp CXX=g++
mkdir -p results
for threads in 1 2 4 8; do
  export OMP_NUM_THREADS="$threads"
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  for rep in 1 2 3; do
    srun -n 1 -c 8 env OMP_NUM_THREADS="$threads" build/poisson-omp --n 96 --iterations 300 --tolerance 0 --output "results/openmp-${threads}-${rep}.json"
  done
done
export OMP_NUM_THREADS=2
srun -n 1 -c 8 env OMP_NUM_THREADS=2 build/poisson-omp --n 24 --iterations 10000 --tolerance 1e-5 --check 50 --output results/precision-double.json
srun -n 1 -c 8 env OMP_NUM_THREADS=2 build/poisson-omp-mixed --n 24 --iterations 10000 --tolerance 1e-5 --check 50 --output results/precision-mixed.json
