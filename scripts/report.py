import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/"results"
groups={}
for kind in ("strong","weak"):
    groups[kind]={}
    for ranks in (1,2,4,8):
        files=sorted(root.glob(f"{kind}-{ranks}-*.json"))
        if files:groups[kind][ranks]=[json.loads(p.read_text())["seconds"] for p in files]
for path in root.glob("correctness-*.json"):
    r=json.loads(path.read_text())
    expected=(1-.8*(1-math.cos(math.pi/14)))**19
    assert math.isclose(r["relative_residual"],expected,rel_tol=1e-10),path
if (root/"gpu-correctness.json").exists():
    r=json.loads((root/"gpu-correctness.json").read_text())
    assert math.isclose(r["relative_residual"],(1-.8*(1-math.cos(math.pi/14)))**19,rel_tol=1e-10)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
lines=["# Measured cluster results","","Fixed-sweep timings. Three repetitions per configuration.","",
       "| Grid | MPI ranks | Median seconds | Speedup / weak efficiency |","|---|---:|---:|---:|"]
for kind,ax in zip(("strong","weak"),axes):
    g=groups[kind]
    if not g:continue
    ranks=sorted(g);med=[statistics.median(g[r]) for r in ranks];base=med[0]
    values=[base/t if kind=="strong" else base/t for t in med]
    ax.plot(ranks,values,"o-",label="Measured")
    ax.plot(ranks,ranks if kind=="strong" else [1]*len(ranks),"--",color="gray",label="Ideal")
    ax.set(xlabel="MPI ranks",ylabel="Speedup" if kind=="strong" else "Weak efficiency",
           title="Strong scaling: 128³" if kind=="strong" else "Weak scaling: 64×64×64 per rank")
    ax.set_xticks(ranks);ax.grid(alpha=.2);ax.legend()
    for r,t,v in zip(ranks,med,values):lines.append(f"| {kind} | {r} | {t:.5f} | {v:.3f} |")
fig.savefig(root/"scaling.png",dpi=160);plt.close(fig)
gpu=[json.loads(p.read_text())["seconds"] for p in root.glob("gpu-[0-9]*.json")]
serial=[json.loads(p.read_text())["seconds"] for p in root.glob("serial-[0-9]*.json")]
if gpu and serial:
    ratio=statistics.median(serial)/statistics.median(gpu)
    lines.extend(["",f"Single-GPU FP64 vs one CPU core: median speedup {ratio:.2f}×.",
                  "This comparison does not represent optimized CPU or GPU library performance."])
else:lines.extend(["","GPU validation and timings are pending; no GPU performance claim is made."])
omp={}
for threads in (1,2,4,8):
    records=[json.loads(p.read_text()) for p in root.glob(f"openmp-{threads}-*.json")]
    assert all(r['threads']==threads for r in records),'Scheduler overrode OMP_NUM_THREADS: rerun with env set inside srun'
    timings=[r['seconds'] for r in records]
    if timings:omp[threads]=statistics.median(timings)
if omp:
    lines.extend(["","## OpenMP threading","","96³ grid, 300 fixed sweeps, one process allocated eight CPU cores; three repetitions.","",
                  "| Threads | Median seconds | Speedup |","|---|---:|---:|"])
    for threads,seconds in omp.items():lines.append(f"| {threads} | {seconds:.5f} | {omp[1]/seconds:.3f} |")
if (root/"precision-mixed.json").exists():
    lines.extend(["","## Storage precision","","24³ grid, two threads, convergence target 1e-5; FP64 update arithmetic in both configurations.","",
                  "| Storage | Sweeps | Relative residual | Relative solution error |","|---|---:|---:|---:|"])
    for name in ("double","mixed"):
        r=json.loads((root/f"precision-{name}.json").read_text())
        assert r['threads']==2
        assert math.isfinite(r["relative_residual"]) and r["relative_residual"]<1e-5
        lines.append(f"| {name} | {r['iterations']} | {r['relative_residual']:.2e} | {r['relative_solution_error']:.2e} |")
    lines.append("These results do not establish that FP32 fields can satisfy arbitrary tighter tolerances.")
lines.extend(["","Raw timings are retained as JSON. These measurements characterize this implementation",
              "and these resource placements; they are not hardware peak-performance estimates.",""])
(root/"REPORT.md").write_text("\n".join(lines))
print("\n".join(lines[:18]))
