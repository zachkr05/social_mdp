# Which object the partner wants, inferred or known: two_rooms scene, w = 1 / 1, cooperation selective, every robot at levels 2 / 2, swaps on a place, taking from hands, Eq. 7's tau 0.3, a move with the partner's object costs 1 x a move, a hinderer without a goal of its own, inertia: no undoing the last action unless it is worth 1 x a move's cost more than the next best, two posteriors by the partner's hands (its object while it holds nothing, its destination while it holds an object)

Transition at which agent_1 / agent_2 finished its own goal (– = never; no goal: a hinderer without a goal of its own); with the objects inferred, each robot's final belief that its partner wants the partner's own object and that it goes to its true destination (– : a level-0 robot holds no belief); acts for the partner that did not count (set-downs / hand-overs, per robot).

| pairing | inferred | belief: partner's object | ... its destination | in vain | wall time |
|---|---|---|---|---|---|
| coercion / cooperation | 16 / 20 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 516 s |
| coercion / conflict | 14 / 23 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 494 s |
| coercion / competition | 13 / 15 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 304 s |
| coercion / coercion | 13 / 23 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 500 s |
| coercion / exchange | 13 / 15 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 296 s |
| coercion / none | 13 / 15 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 274 s |
| coercion / hinder | 40 / no goal | 0.90 / 1.00 | 0.17 / 1.00 | 0+0 / 0+0 | 1220 s |
