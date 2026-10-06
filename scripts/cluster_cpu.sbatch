#!/bin/bash
#SBATCH --job-name=poisson-cpu
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/cpu-%j.out
set -euo pipefail
module load cesga/2025 gompi/2025.gcc13
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
make mpi
mkdir -p results
hostname > "results/cpu-hardware-${SLURM_JOB_ID}.txt"
lscpu >> "results/cpu-hardware-${SLURM_JOB_ID}.txt"
module -t list 2>> "results/cpu-hardware-${SLURM_JOB_ID}.txt"
for ranks in 1 2 4 8; do
  for repetition in 1 2 3; do
    mpirun -np "$ranks" ./build/poisson-mpi --n 128 --iterations 500 --tolerance 0 --overlap \
      --output "results/strong-${ranks}-${repetition}.json"
    mpirun -np "$ranks" ./build/poisson-mpi --nx 64 --ny 64 --nz "$((64*ranks))" \
      --iterations 500 --tolerance 0 --overlap --output "results/weak-${ranks}-${repetition}.json"
  done
done
for ranks in 1 2 3; do
  mpirun -np "$ranks" ./build/poisson-mpi --n 13 --iterations 19 --tolerance 0 --overlap \
    --output "results/correctness-${ranks}.json"
done
