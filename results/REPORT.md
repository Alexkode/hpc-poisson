# Measured cluster results

Fixed-sweep timings. Three repetitions per configuration.

| Grid | MPI ranks | Median seconds | Speedup / weak efficiency |
|---|---:|---:|---:|
| strong | 1 | 2.40438 | 1.000 |
| strong | 2 | 1.29937 | 1.850 |
| strong | 4 | 0.98059 | 2.452 |
| strong | 8 | 0.38311 | 6.276 |
| weak | 1 | 0.22447 | 1.000 |
| weak | 2 | 0.24519 | 0.916 |
| weak | 4 | 0.33290 | 0.674 |
| weak | 8 | 0.37252 | 0.603 |

Single-GPU FP64 vs one CPU core: median speedup 94.95×.
This comparison does not represent optimized CPU or GPU library performance.

## OpenMP threading

96³ grid, 300 fixed sweeps, one process allocated eight CPU cores; three repetitions.

| Threads | Median seconds | Speedup |
|---|---:|---:|
| 1 | 0.61316 | 1.000 |
| 2 | 0.24321 | 2.521 |
| 4 | 0.14943 | 4.103 |
| 8 | 0.06981 | 8.784 |

## Storage precision

24³ grid, two threads, convergence target 1e-5; FP64 update arithmetic in both configurations.

| Storage | Sweeps | Relative residual | Relative solution error |
|---|---:|---:|---:|
| double | 1850 | 8.23e-06 | 1.31e-03 |
| mixed | 1850 | 9.49e-06 | 1.31e-03 |
These results do not establish that FP32 fields can satisfy arbitrary tighter tolerances.

Raw timings are retained as JSON. These measurements characterize this implementation
and these resource placements; they are not hardware peak-performance estimates.
