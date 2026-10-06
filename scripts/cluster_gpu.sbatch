#!/bin/bash
#SBATCH --job-name=poisson-gpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/gpu-%j.out
set -euo pipefail
module load cesga/2025 gcc/13.4.0 cuda/12.9.1
cd "${SLURM_SUBMIT_DIR}"
make cuda
make CXX=g++
mkdir -p results
nvidia-smi > "results/gpu-hardware-${SLURM_JOB_ID}.txt"
for repetition in 1 2 3; do
  ./build/poisson-cuda --n 128 --iterations 500 --tolerance 0 \
    --output "results/gpu-${repetition}.json"
  ./build/poisson --n 128 --iterations 500 --tolerance 0 \
    --output "results/serial-${repetition}.json"
done
./build/poisson-cuda --n 13 --iterations 19 --tolerance 0 --output results/gpu-correctness.json
