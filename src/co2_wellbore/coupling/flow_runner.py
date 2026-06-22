from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple


def run_flow(
    flow_exe: str,
    case_root: str,
    cwd: Path,
    timeout_sec: float,
    mpi_nproc: int,
    flow_args: Optional[List[str]] = None,
) -> Tuple[bool, str, float]:
    """Run OPM Flow with optional extra command-line arguments."""
    extra = list(flow_args or [])

    if int(mpi_nproc) > 1:
        cmd = ["mpirun", "--bind-to", "core", "-np", str(int(mpi_nproc)), flow_exe] + extra + [case_root]
    else:
        cmd = [flow_exe] + extra + [case_root]

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    t0 = time.time()
    with (cwd / "FLOW_STDOUT.log").open("w", encoding="utf-8") as f:
        f.write("FLOW COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=f, text=True, env=env)
        try:
            rc = proc.wait(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "timeout", time.time() - t0

    dt = time.time() - t0
    return (rc == 0), ("ok" if rc == 0 else f"returncode_{rc}"), dt