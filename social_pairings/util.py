"""Small helpers shared by the solver and the rollout."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numba
import numpy as np

from .constants import (CHUNK, GIVE, N_ACTIONS, PICK, PLACE, SWAP, TAKE, TIE_REL, WAIT)

__all__ = ["configure_threads", "check", "log_softmax", "normalize_log", "argmax_action",
           "sample_action", "stay_table", "block_items", "top_hypotheses", "memory_limit_bytes"]


def _available_cpus() -> int:
    try:
        n = len(os.sched_getaffinity(0))          # respects cgroups / cpusets (SLURM)
    except AttributeError:
        n = os.cpu_count() or 1
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "NSLOTS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            n = min(n, int(v))
            break
    return max(1, n)


def memory_limit_bytes() -> Optional[int]:
    """The memory this process may use: the machine's, or the limit of its control group
    (a container, a SLURM job) where that is lower.  None where neither can be read."""
    sizes = []
    try:
        sizes.append(int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES")))
    except (AttributeError, ValueError, OSError):
        pass
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:                   # "<id>:<controllers>:<path>"; v2: "0::<path>"
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, where = parts[1], parts[2].lstrip("/")
        if controllers == "":
            files = (Path("/sys/fs/cgroup") / where / "memory.max",)
        elif "memory" in controllers.split(","):
            files = (Path("/sys/fs/cgroup/memory") / where / "memory.limit_in_bytes",)
        else:
            continue
        for f in files:
            try:
                text = f.read_text().strip()
            except OSError:
                continue
            if text.isdigit() and int(text) > 0:
                sizes.append(int(text))
    return min(sizes) if sizes else None


def configure_threads(requested: int = 0) -> int:
    """Use every CPU this process may use (or ``requested``) for the numba kernels."""
    n = requested if requested > 0 else _available_cpus()
    n = max(1, min(n, numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    return n


def check(ok, msg: str) -> None:
    """Invariant check that survives ``python -O`` (unlike assert)."""
    if not ok:
        raise AssertionError(f"[pairings] invariant violated: {msg}")


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis; -inf (illegal) entries stay -inf."""
    y = z - np.max(z, axis=-1, keepdims=True)
    return y - np.log(np.sum(np.exp(y), axis=-1, keepdims=True))


def normalize_log(logp: np.ndarray) -> np.ndarray:
    m = float(np.max(logp))
    return logp - (m + np.log(np.sum(np.exp(logp - m))))


def argmax_action(q: np.ndarray) -> int:
    """Algorithm 1's argmax with a deterministic tie-break: the lowest action index among
    the actions whose Q is within round-off (1e-9 relative) of the maximum, so that exact
    ties on symmetric grids do not depend on solver round-off or the thread count."""
    m = float(np.max(q))
    return int(np.flatnonzero(q >= m - TIE_REL * max(1.0, abs(m)))[0])


# the move that undoes each move (action indices 0..3: +y, +x, -y, -x)
OPPOSITE_MOVE = (2, 3, 0, 1)
# the hand action that undoes each hand action: place what it picked up, pick up what it
# placed, give back what it took, take back what it gave, swap back
OPPOSITE_HAND = {PICK: PLACE, PLACE: PICK, TAKE: GIVE, GIVE: TAKE, SWAP: SWAP}


def undo_of(action: int, moved: bool, hands_changed: bool) -> Optional[int]:
    """The action that would undo ``action`` right after it (``Config.inertia``): the
    opposite move after a move that took the robot to another cell; after a hand action that
    changed what it holds, the hand action that changes it back (``OPPOSITE_HAND``: placing
    what it picked up, picking up what it placed, giving back what it took, taking back what
    it gave, swapping back); None after a wait, a blocked move or a hand action that changed
    nothing."""
    if action < WAIT:
        return OPPOSITE_MOVE[action] if moved else None
    if action >= PICK and hands_changed:
        return OPPOSITE_HAND[action]
    return None


def choose_with_inertia(q: np.ndarray, undo: Optional[int], margin: float
                        ) -> Tuple[int, bool]:
    """Algorithm 1's argmax with inertia: the robot does not undo its previous action
    unless undoing beats every other action by more than ``margin``.

    q (10,) : Q of each action, -inf where illegal.  undo : the action that would undo the
    previous one (:func:`undo_of`), or None.  margin : ``Config.inertia_margin``.
    -> (action, True where inertia changed the choice).  With margin 0 or nothing to undo,
    :func:`argmax_action` exactly."""
    best = argmax_action(q)
    if undo is None or margin <= 0.0 or best != undo:
        return best, False
    rest = np.array(q, dtype=float, copy=True)
    rest[undo] = -np.inf
    if not np.isfinite(rest).any():          # undoing is all it can do
        return best, False
    other = argmax_action(rest)
    if float(q[undo]) - float(q[other]) > margin:
        return best, False
    return other, True


def sample_action(pi: np.ndarray, u: float) -> int:
    """Inverse-CDF draw from the distribution ``pi`` with the uniform ``u``."""
    c = np.cumsum(pi)
    a = int(np.searchsorted(c, u * c[-1], side="right"))
    if a >= pi.size or pi[a] <= 0.0:      # round-off at the top of the CDF
        a = int(np.flatnonzero(pi > 0.0)[-1])
    return a


def stay_table(valid: np.ndarray, eps: float) -> np.ndarray:
    """(N, 10) action distribution of a robot that stays put: "stay" w.p. 1 - eps, eps spread
    evenly over its legal moves ("stay" w.p. 1 on a cell with no legal move); never a hand
    action."""
    moves = valid[:, :WAIT].astype(float)                                # (N, 4)
    nm = moves.sum(axis=1)                                               # (N,)
    out = np.zeros((valid.shape[0], N_ACTIONS), dtype=float)
    out[:, :WAIT] = moves * (eps / np.maximum(nm, 1.0))[:, None]
    out[:, WAIT] = np.where(nm > 0, 1.0 - eps, 1.0)
    return np.ascontiguousarray(out)


def block_items(NN: int, blocks, label: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Work items ``(label, lo, hi)`` covering the given blocks in CHUNK-state pieces."""
    los, his = [], []
    for b in np.asarray(blocks).tolist():
        st = np.arange(b * NN, (b + 1) * NN, CHUNK, dtype=np.int64)
        los.append(st)
        his.append(np.minimum(st + CHUNK, (b + 1) * NN))
    if not los:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e
    lo, hi = np.concatenate(los), np.concatenate(his)
    return np.full(lo.size, label, dtype=np.int64), lo, hi


def top_hypotheses(b: np.ndarray, truth: Optional[int], k: int) -> str:
    """The k most likely destinations of a belief, the true one starred: "g0*=0.91 g2=0.05"."""
    idx = np.argsort(-b, kind="stable")[:k]
    return " ".join(f"g{j}{'*' if j == truth else ''}={b[j]:.2f}" for j in idx)
