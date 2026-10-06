# Experiment provenance

Experiments recorded on 2026-10-06. Raw configurations, seeds, timings and data hashes are published in `results/`.

| Experiment | Hardware / software | Boundary |
|---|---|---|
| Poisson MPI | CESGA FT3, Intel Xeon Platinum 8352Y; GCC 13.4, Open MPI 5.0.9 | 1/2/4/8 ranks, three repetitions, fixed sweeps |
| Poisson OpenMP / storage precision | CESGA CPU, GCC 13.4 | 1/2/4/8 threads, 96³ fixed workload; FP64/FP32 storage checked at 24³ and tolerance 1e-5 |
| Poisson CUDA | CESGA, NVIDIA A100 PCIe 40GB; CUDA compiler 12.9 | One GPU, FP64 resident solution, fixed sweeps; analytical correctness separately verified |

These are the measured configurations. Optional larger or GPU configurations in scripts are not additional measured results. Timings depend on hardware and software; consult `results/REPORT.md` for protocol details and limitations.

## Repository history

This standalone repository was extracted from the original project directory on 2026-10-06. Existing author and commit timestamps were preserved. The original directory contents were verified against the extracted Git tree before adding standalone documentation and checks. Code was developed with AI assistance.
