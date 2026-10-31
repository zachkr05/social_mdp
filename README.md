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
  `run_spare2.sh` -- one pairing with two spare objects (needs ~190 GiB). See `RUN_ALL_49.md`.
- `tools/try_pairing.py` -- a pairing's beliefs, transition by transition.

```
pip install -r requirements.txt
python -m pytest social_pairings/tests/test_beliefs.py      # the belief update, < 1 min
TAU=0.15 sh run_all_49_parallel.sh                           # the 49 pairings
```
