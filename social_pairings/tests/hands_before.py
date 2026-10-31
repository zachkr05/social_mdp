"""The rule of the hands until 2026-10-18, for the tests written under it.

Until 2026-10-18 a robot took its own object on arriving at the cell where it lay and set
it down on arriving at its destination, both without a hand action and for free; a pick-up
was instant and cost a move, and so did every other hand action
(``social_pairings.config.AUTOMATIC_HANDS``).  Since then nothing is automatic (the user:
"nothing is automatic ... you must move to the cell then select the pickup dropoff etc
action"), a pick-up takes three transitions and costs three moves' worth on the first, and
the other hand actions cost two (``Config.automatic_hands``, ``pick_steps``,
``pick_cost_rate``, ``use_cost_rate``).  The tests written before pin results made under the
rule of then, and keep doing so through these stand-ins: the package's functions with
``AUTOMATIC_HANDS`` for defaults.  An option a test passes overrides it.  The rule of now
is tested in ``test_explicit.py``, which imports from the package.  (``told.py`` is the same
for the defaults of before 2026-10-08, the rule of the hands of then included.)
"""
from social_pairings import AUTOMATIC_HANDS
from social_pairings import Config as _Config
from social_pairings import default_levels as _default_levels
from social_pairings import levels_used as _levels_used
from social_pairings import run_baseline as _run_baseline
from social_pairings import run_pairing as _run_pairing

__all__ = ["Config", "default_levels", "levels_used", "run_baseline", "run_pairing"]


def Config(**options) -> _Config:  # noqa: N802 -- stands in for the class
    """``social_pairings.Config`` with the rule of the hands until 2026-10-18."""
    return _Config(**{**AUTOMATIC_HANDS, **options})


def run_pairing(scene, goals, *, cfg=None, **kwargs) -> dict:
    return _run_pairing(scene, goals, cfg=Config() if cfg is None else cfg, **kwargs)


def run_baseline(scene, *, cfg=None, **kwargs) -> dict:
    return _run_baseline(scene, cfg=Config() if cfg is None else cfg, **kwargs)


def default_levels(goals, cfg=None, weights=None) -> tuple:
    return _default_levels(goals, Config() if cfg is None else cfg, weights)


def levels_used(goals, levels, cfg=None, weights=None) -> tuple:
    return _levels_used(goals, levels, Config() if cfg is None else cfg, weights)
