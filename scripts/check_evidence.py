"""Validate the configurations in the published experimental records."""
import json
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text())

hpc=root/'results'
for ranks in [1,2,4,8]:
    for kind in ['strong','weak']:
        records=[read(p) for p in hpc.glob(f'{kind}-{ranks}-*.json')]
        assert len(records)==3
        assert all(r['ranks']==ranks and r['iterations']==500 and r['seconds']>0 for r in records)
for threads in [1,2,4,8]:
    records=[read(p) for p in hpc.glob(f'openmp-{threads}-*.json')]
    assert len(records)==3 and all(r['threads']==threads for r in records)
for name in ['double','mixed']:
    r=read(hpc/f'precision-{name}.json');assert r['threads']==2 and r['relative_residual']<1e-5
for p in hpc.glob('gpu-[0-9]*.json'):
    r=read(p);assert r['nx']==128 and r['iterations']==500 and r['backend']=='cuda'

print('Published evidence checks passed for hpc-poisson.')
