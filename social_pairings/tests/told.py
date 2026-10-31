"""The defaults the tests written before 2026-10-08 ran under: each robot told its partner's
social goal.

Until 2026-10-08 ``Config``'s defaults told each level-2 robot its partner's social goal, took
an exchanger at level 1 as Table I prints it and read competition as "race_then_hinder"
(``social_pairings.config.TOLD``).  The tests written then pin results made under those
defaults, and keep doing so through these stand-ins: the package's functions with ``TOLD``
for defaults.  An option a test passes overrides it as before.  The defaults of now (every
partner goal inferred) are tested in ``test_defaults.py``, which imports from the package.

Also until 2026-10-08 a robot with a physical goal only reasoned at level 2 when no level
was asked for (``physical_level=2``), and no robot held its partner possibly idle
(``idle_hypothesis=False``); the stand-ins keep those as well (``BEFORE``).  ``TOLD`` itself
keeps what changed later: each robot knowing which object its partner wants (until
2026-10-09), Table I's cooperator and a hinderer at level 1 (until 2026-10-11).
"""
from social_pairings import TOLD
from social_pairings import Config as _Config
from social_pairings import default_levels as _default_levels
from social_pairings import levels_used as _levels_used
from social_pairings import run_baseline as _run_baseline
from social_pairings import run_pairing as _run_pairing

__all__ = ["BEFORE", "Config", "default_levels", "levels_used", "run_baseline", "run_pairing"]

BEFORE = {**TOLD, "physical_level": 2, "idle_hypothesis": False}


def Config(**options) -> _Config:  # noqa: N802 -- stands in for the class
    """``social_pairings.Config`` with the defaults of before 2026-10-08."""
    return _Config(**{**BEFORE, **options})


def run_pairing(scene, goals, *, cfg=None, **kwargs) -> dict:
    return _run_pairing(scene, goals, cfg=Config() if cfg is None else cfg, **kwargs)


def run_baseline(scene, *, cfg=None, **kwargs) -> dict:
    return _run_baseline(scene, cfg=Config() if cfg is None else cfg, **kwargs)


def default_levels(goals, cfg=None, weights=None) -> tuple:
    return _default_levels(goals, Config() if cfg is None else cfg, weights)


def levels_used(goals, levels, cfg=None, weights=None) -> tuple:
    return _levels_used(goals, levels, Config() if cfg is None else cfg, weights)
