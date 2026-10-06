"""Collect actual runs with compiler/hardware provenance, never simulated timings."""
import argparse
import json
import platform
import subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",default="build/poisson")
    p.add_argument("--n",type=int,default=96)
    p.add_argument("--iterations",type=int,default=500)
    p.add_argument("--repetitions",type=int,default=5)
    p.add_argument("--output",default="results/local.json")
    a=p.parse_args()
    command=[a.binary,"--n",str(a.n),"--iterations",str(a.iterations),"--tolerance","0"]
    rows=[json.loads(subprocess.check_output(command,text=True)) for _ in range(a.repetitions)]
    result={"hardware":platform.platform(),"machine":platform.machine(),"command":command,"runs":rows}
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(target),"runs":len(rows)}))

if __name__=="__main__":main()
