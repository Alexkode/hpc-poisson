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

Raw timings are retained as JSON. These measurements characterize this implementation
and these resource placements; they are not hardware peak-performance estimates.
