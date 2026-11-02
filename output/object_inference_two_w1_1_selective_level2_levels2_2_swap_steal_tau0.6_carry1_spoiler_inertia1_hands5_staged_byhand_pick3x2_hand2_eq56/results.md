# Which object the partner wants, inferred or known: two_rooms scene, w = 1 / 1, cooperation selective, every robot at levels 2 / 2, swaps on a place, taking from hands, Eq. 7's tau 0.6, a move with the partner's object costs 1 x a move, a hinderer without a goal of its own, inertia: no undoing the last action unless it is worth 1 x a move's cost more than the next best, two posteriors by the partner's hands (its object while it holds nothing, its destination while it holds an object)

Transition at which agent_1 / agent_2 finished its own goal (– = never; no goal: a hinderer without a goal of its own); with the objects inferred, each robot's final belief that its partner wants the partner's own object and that it goes to its true destination (– : a level-0 robot holds no belief); acts for the partner that did not count (set-downs / hand-overs, per robot).

| pairing | inferred | belief: partner's object | ... its destination | in vain | wall time |
|---|---|---|---|---|---|
| exchange / cooperation | 16 / 20 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 534 s |
| exchange / conflict | 13 / 25 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 475 s |
| exchange / competition | 13 / 15 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 318 s |
| exchange / coercion | 13 / 23 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 486 s |
| exchange / exchange | 13 / 18 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 388 s |
| exchange / none | 13 / 15 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 284 s |
| exchange / hinder | 35 / no goal | 0.99 / 1.00 | 0.12 / 1.00 | 0+0 / 0+0 | 1086 s |
