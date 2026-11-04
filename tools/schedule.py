"""Run the rollouts of a jobs file as many at a time as the machine's memory and CPUs allow,
each scene with its own memory and thread needs (2026-11-04; the user: "parallelize the
S-MDP so it runs substantially faster").

Until now a grid ran two fixed pools (run_testbed.sh: the gate scene's few big rollouts in
one, the house scenes' many small ones in another), so that when one pool ran out of
work its share of the machine sat idle -- the gate pool, sized for the memory the house
pool left it, would have run its 28 rollouts three at a time for days after the house
was done.  Here there is one queue, the long rollouts first, and a job starts whenever
its memory and threads fit: the long ones take the machine at the start, the short ones
fill in around them, and the whole machine stays busy to the end.

    python3 tools/schedule.py JOBS --logs LOGS --flags "..." --extra "..." \
        --mem 18 --mem gate=72 --threads 2 --threads gate=4 --max-threads 8

JOBS: one rollout per line, "scene tau1 tau2 goal1 goal2" (run_tau_grid.sh writes it).
--mem GiB a rollout of a scene needs (a default and per-scene overrides); the jobs
running at once never claim more than 90 % of the machine's memory (--ram to override the
machine's).  --threads: a scene's threads (default and overrides); a job gets more when
CPUs are free -- the free CPUs shared out over the jobs that still fit in memory, up to
--max-threads.  Each job runs examples/run_object_inference.py with the given flags,
--resume, one log per rollout in LOGS/run_<scene>_<tau1>_<tau2>_<goal1>_<goal2>.log; what
fails is retried once at the end and listed in LOGS/failed.txt.  NUMA=1 in the environment
(with numactl installed) binds each job to one NUMA node in turn.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    scene: str
    t1: str
    t2: str
    a: str
    b: str
    line: str
    mem: float = 0.0
    threads: int = 0
    proc: subprocess.Popen | None = None
    log: Path | None = None
    started: float = 0.0
    tries: int = 0

    @property
    def name(self) -> str:
        return f"{self.scene}_{self.t1}_{self.t2}_{self.a}_{self.b}"

    def describe(self) -> str:
        return f"{self.scene}, agent_1 at tau {self.t1}, agent_2 at {self.t2}, {self.a} / {self.b}"


def per_scene(values: list[str], default: float) -> tuple[float, dict]:
    """--mem 18 --mem gate=72 -> (18, {"gate": 72})."""
    over = {}
    for v in values:
        if "=" in v:
            k, x = v.split("=", 1)
            over[k] = float(x)
        else:
            default = float(v)
    return default, over


def machine() -> tuple[int, float]:
    cpus = os.cpu_count() or 2
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError):
        ram = 16.0
    # a control group may allow less
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
            if text.isdigit():
                ram = min(ram, int(text) / 2 ** 30)
        except OSError:
            pass
    return cpus, ram


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--flags", default="", help="run_object_inference.py's model flags")
    ap.add_argument("--extra", default="", help="... and its other flags (--steps, --no-video, ...)")
    ap.add_argument("--mem", action="append", default=[], metavar="[SCENE=]GIB")
    ap.add_argument("--threads", action="append", default=[], metavar="[SCENE=]N")
    ap.add_argument("--max-threads", type=int, default=8)
    ap.add_argument("--ram", type=float, default=None, help="GiB to plan with (default: the machine's)")
    ap.add_argument("--cpus", type=int, default=None)
    ap.add_argument("--share", type=float, default=0.9, help="share of the memory the jobs may claim")
    ap.add_argument("--python", default=sys.executable or "python3")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between checks")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    cpus, ram = machine()
    if args.cpus:
        cpus = args.cpus
    if args.ram:
        ram = args.ram
    budget = args.share * ram
    mem_default, mem_over = per_scene(args.mem, 2.5)
    thr_default, thr_over = per_scene(args.threads, 2)
    jobs: list[Job] = []
    for line in args.jobs.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        j = Job(*parts, line=line.strip())
        j.mem = float(mem_over.get(j.scene, mem_default))
        j.threads = int(thr_over.get(j.scene, thr_default))
        jobs.append(j)
    if not jobs:
        print("schedule: no jobs")
        return 0
    # the long rollouts first (by memory, a proxy for size), the order of the file otherwise
    order = sorted(range(len(jobs)), key=lambda i: (-jobs[i].mem, i))
    queue = [jobs[i] for i in order]
    for j in queue:
        if j.mem > budget:
            print(f"schedule: {j.describe()} needs {j.mem:g} GiB, more than the {budget:.0f} GiB "
                  f"available: it will run alone", flush=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    failed_path = args.logs / "failed.txt"
    failed_path.write_text("")
    numa_nodes = 0
    if os.environ.get("NUMA", "0") == "1" and shutil.which("numactl"):
        try:
            out = subprocess.run(["numactl", "-H"], capture_output=True, text=True, check=True).stdout
            numa_nodes = int(out.split("\n")[0].split()[1])
        except (subprocess.CalledProcessError, ValueError, IndexError):
            numa_nodes = 0
    print(f"schedule: {len(jobs)} rollouts; {ram:.0f} GiB ({budget:.0f} to use), {cpus} CPUs"
          + (f", {numa_nodes} NUMA nodes" if numa_nodes > 1 else ""), flush=True)
    for scene in dict.fromkeys(j.scene for j in queue):
        n = sum(1 for j in queue if j.scene == scene)
        j = next(j for j in queue if j.scene == scene)
        print(f"  {scene}: {n} rollouts, {j.mem:g} GiB and {j.threads} threads each", flush=True)

    running: list[Job] = []
    done = failed = 0
    launched = 0
    t0 = time.time()

    def free() -> tuple[float, int]:
        return (budget - sum(j.mem for j in running), cpus - sum(j.threads for j in running))

    def launch(j: Job, threads: int) -> None:
        nonlocal launched
        j.threads = threads
        j.tries += 1
        j.log = args.logs / f"run_{j.name}.log"
        cmd = ([args.python, "examples/run_object_inference.py"] + args.flags.split()
               + args.extra.split() + ["--scene", j.scene, "--temperature", j.t1, j.t2, "--threads",
                                       str(threads), "--max-memory-gib", f"{j.mem:g}", "--resume",
                                       "--pairing", j.a, j.b])
        if numa_nodes > 1:
            node = launched % numa_nodes
            cmd = ["numactl", f"--cpunodebind={node}", f"--preferred={node}"] + cmd
        env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(j.log, "ab") as out:
            j.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      env=env)
        j.started = time.time()
        running.append(j)
        launched += 1
        print(f"start: {j.describe()} -- {threads} threads, {j.mem:g} GiB "
              f"({len(running)} running, {len(queue)} waiting)", flush=True)

    def reap() -> None:
        nonlocal done, failed
        for j in list(running):
            code = j.proc.poll()
            if code is None:
                continue
            running.remove(j)
            took = (time.time() - j.started) / 3600
            if code == 0:
                done += 1
                print(f"done: {j.describe()} -- {took:.1f} h ({done} done, {len(running)} running, "
                      f"{len(queue)} waiting)", flush=True)
            elif j.tries <= args.retries:
                print(f"FAILED (will retry): {j.describe()} after {took:.1f} h (see {j.log})", flush=True)
                queue.append(j)                  # once more, after the others
            else:
                failed += 1
                print(f"FAILED: {j.describe()} after {took:.1f} h (see {j.log})", flush=True)
                with open(failed_path, "a") as f:
                    f.write(j.line + "\n")

    while queue or running:
        reap()
        progress = True
        while progress and queue:
            progress = False
            mem_free, cpu_free = free()
            for j in list(queue):
                if j.mem > mem_free and running:
                    continue                     # does not fit now (alone, it always runs)
                if j.threads > cpu_free and running:
                    continue
                # the free CPUs shared out over as many such jobs as still fit in memory
                slots = max(1, min(int(mem_free // j.mem), sum(1 for k in queue if k.mem == j.mem)))
                threads = max(j.threads, min(args.max_threads, cpu_free // slots))
                if running:
                    threads = max(1, min(threads, cpu_free))
                queue.remove(j)
                launch(j, threads)
                progress = True
                break
        if queue or running:
            time.sleep(args.poll)
    hours = (time.time() - t0) / 3600
    print(f"schedule: {done} done, {failed} failed, {hours:.1f} h", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
