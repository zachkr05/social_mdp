# social_pairings

A two-robot pick-and-place replication of the Social MDP of Tejwani et al. ("Social
Interactions as Recursive MDPs", CoRL 2021; "Incorporating Rich Social Interactions Into
MDPs", ICRA 2022), checked against the authors' solver
([Social-MDP/social-mdp-framework](https://github.com/Social-MDP/social-mdp-framework),
branch `social-interactions`).

- `social_pairings/` -- the package: world, Social MDP solver (levels 0-2), the belief update
  of Eqs. 5-6, rollouts, figures and videos. Its `README.md` documents the model, every option
  and every result.
- `examples/` -- `run_object_inference.py` (one or all pairings, records and videos) and the
  other experiment scripts.
- `run_all_49.sh`, `run_all_49_parallel.sh` -- the 49 pairings at the standard settings;
  `run_tau_grid.sh` -- the 49 at every pair of taus, each robot its own;
  `run_spare2.sh` -- one pairing with two spare objects (needs ~190 GiB). See `RUN_ALL_49.md`.
- `tools/try_pairing.py` -- a pairing's beliefs, transition by transition; `tools/tau_grid.py`
  -- the tau grid's figures and tables.

```
pip install -r requirements.txt
python -m pytest social_pairings/tests/test_beliefs.py      # the belief update, < 1 min
TAU=0.15 sh run_all_49_parallel.sh                           # the 49 pairings
TAU="0.2 0.5" sh run_all_49_parallel.sh                      # ... agent_1 at tau 0.2, agent_2 at 0.5
sh run_tau_grid.sh                                           # ... at every pair of taus 0.1-0.5
CONFLICT=task sh run_all_49_parallel.sh                      # ... with the task conflict, -w r(g_j)
SAME=1 TAUS="0.1 0.3 0.5" HAZARD=0.02 sh run_tau_grid.sh     # ... the partner's goal may change
```

Since 2026-11-02 the partner's goal may change, as an option (off by default): `HAZARD=h`
mixes the belief about it with the uniform after every update, so that a crushed hypothesis
can come back (Baker, Saxe & Tenenbaum 2009's model M2; `social_pairings/README.md`, "The
partner's goal may change").

Conflict is Table I's, $-w\,\tilde R_j$ above level 1. Since 2026-11-01 (evening) the task
conflict, $R_i = r(g_i) - w\,r(\tilde g_j) - c(a_i)$ against every partner, is an option:
`Config.conflict="task"`, `--conflict task`, `CONFLICT=task` (`social_pairings/README.md`,
"Task conflict"; `RUN_ALL_49.md`).
