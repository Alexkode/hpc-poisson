"""Independent analytic invariants, including uneven MPI domain decomposition."""
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(n=12, iterations=10000, tolerance=1e-9, binary="poisson", ranks=None, extra=()):
    command = [str(ROOT / "build" / binary), "--n", str(n), "--iterations", str(iterations),
               "--tolerance", str(tolerance), *extra]
    if ranks:
        command = ["mpirun", "--oversubscribe", "-np", str(ranks), *command]
    return json.loads(subprocess.check_output(command, env={**os.environ,"OMP_NUM_THREADS":"1"},text=True))

def test_manufactured_mode_after_fixed_sweeps():
    n, steps = 13, 19
    result = run(n,steps,0)
    # The sine mode is an eigenvector of the discrete Laplacian.
    factor = 1 - .8 * (1-math.cos(math.pi/(n+1)))
    assert math.isclose(result["relative_residual"],factor**steps,rel_tol=1e-10)

def test_second_order_grid_convergence():
    coarse, fine = run(8), run(17)
    assert coarse["relative_residual"] < 1e-9
    assert fine["relative_residual"] < 1e-9
    ratio = coarse["relative_solution_error"]/fine["relative_solution_error"]
    assert 3.9 < ratio < 4.1

def test_reject_invalid_configuration():
    assert subprocess.run([str(ROOT/"build/poisson"),"--n","0"],capture_output=True).returncode != 0

def test_mpi_uneven_decomposition_and_overlap():
    if not (ROOT/"build/poisson-mpi").exists():
        import pytest
        pytest.skip("MPI binary unavailable")
    reference = run(13,19,0)
    for ranks in (2,3):
        for extra in ((),("--overlap",)):
            result = run(13,19,0,"poisson-mpi",ranks,extra)
            assert math.isclose(reference["relative_residual"],result["relative_residual"],rel_tol=1e-12)
            assert math.isclose(reference["relative_solution_error"],result["relative_solution_error"],rel_tol=1e-12)

def test_cuda_matches_analytic_solution():
    if not os.environ.get("TEST_CUDA"):
        import pytest
        pytest.skip("Set TEST_CUDA=1 on an allocated GPU")
    result = run(13,19,0,"poisson-cuda")
    factor = 1 - .8 * (1-math.cos(math.pi/14))
    assert math.isclose(result["relative_residual"],factor**19,rel_tol=1e-10)
