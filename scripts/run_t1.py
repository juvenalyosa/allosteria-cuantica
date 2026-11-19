"""Run test-1 MOPAC jobs: python run_t1.py MOPAC WORK_DIR JOB[,JOB...] [PARALLEL] [TIMEOUT_S]

Each job directory holds job.mop and env.json (extra environment); finished jobs (job.aux present and a
status.json) are skipped. status.json: wall time, exit code, SCF status lines, MOPAC's error messages.
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def messages(out):
    """The messages MOPAC lists in the box at the end of job.out when a job did not end normally."""
    i = out.rfind("Error and normal termination messages")
    if i < 0:
        return []
    msgs = []
    for ln in out[i:].splitlines()[1:]:
        s = ln.strip().strip("*").strip()
        if s.startswith("JOB ENDED NORMALLY"):
            break
        if s:
            msgs.append(s)
    return msgs[:5]


def run_job(mopac, d, timeout):
    st = d / "status.json"
    if st.exists() and (d / "job.aux").exists():
        res = json.loads(st.read_text())
        if res.get("heat") is None and (d / "job.out").exists():
            res["errors"] = messages((d / "job.out").read_text(errors="ignore")) or res["errors"]
            st.write_text(json.dumps(res, indent=1))
        return res
    env = dict(os.environ)
    if (d / "env.json").exists():
        env.update(json.loads((d / "env.json").read_text()))
    t0 = time.time()
    try:
        p = subprocess.run([mopac, "job.mop"], cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        code, log = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        code, log = "timeout", ""
    out = (d / "job.out").read_text(errors="ignore") if (d / "job.out").exists() else ""
    res = {"job": d.name, "seconds": round(time.time() - t0, 1), "exit": code,
           "heat": (re.findall(r"FINAL HEAT OF FORMATION =\s+(-?\d+\.\d+)", out) or [None])[-1],
           "scf": re.findall(r"\[MOZYME GPU SCF\] status=\S+.*", log + out)[-3:],
           "errors": messages(out) or [ln.strip() for ln in out.splitlines() if "ERROR" in ln.upper()][:5]}
    st.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    mopac, wd = sys.argv[1], Path(sys.argv[2])
    jobs = [wd / j for j in sys.argv[3].split(",")]
    par = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    timeout = float(sys.argv[5]) if len(sys.argv) > 5 else None
    with ThreadPoolExecutor(par) as ex:
        for r in ex.map(lambda d: run_job(mopac, d, timeout), jobs):
            print(json.dumps(r), flush=True)
