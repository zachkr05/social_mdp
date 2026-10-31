"""Results to and from JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["to_json", "save_result", "load_result", "action_names", "USE_ACTIONS"]

# the actions of a record written before 2026-10-16: four moves, stay and "use"
USE_ACTIONS = ("+y", "+x", "-y", "-x", "stay", "use")


def to_json(value):
    """A result record (or any part of one) as plain JSON types: arrays become lists, numpy
    scalars become Python numbers, and non-finite floats (an illegal action's Q) become null."""
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_result(path: Union[str, Path], result: dict) -> Path:
    """Write a result record of :func:`social_pairings.run_pairing` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(result), indent=1) + "\n")
    return path


def load_result(path: Union[str, Path]) -> dict:
    """Read a record written by :func:`save_result` (lists stay lists, null stays None)."""
    return json.loads(Path(path).read_text())


def action_names(result: dict) -> tuple:
    """The actions a record was rolled out with, by number: its ``actions`` (since
    2026-10-16 ``constants.ACTION_NAMES``: four moves, stay and the five hand actions), or
    :data:`USE_ACTIONS` for a record written before, which has none."""
    return tuple(result.get("actions") or USE_ACTIONS)
