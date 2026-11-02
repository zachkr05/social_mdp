# Which object the partner wants, inferred or known: two_rooms scene, w = 1 / 1, cooperation selective, every robot at levels 2 / 2, swaps on a place, taking from hands, Eq. 7's tau 0.2, a move with the partner's object costs 1 x a move, a hinderer without a goal of its own, inertia: no undoing the last action unless it is worth 1 x a move's cost more than the next best, two posteriors by the partner's hands (its object while it holds nothing, its destination while it holds an object)

Transition at which agent_1 / agent_2 finished its own goal (– = never; no goal: a hinderer without a goal of its own); with the objects inferred, each robot's final belief that its partner wants the partner's own object and that it goes to its true destination (– : a level-0 robot holds no belief); acts for the partner that did not count (set-downs / hand-overs, per robot).

| pairing | inferred | belief: partner's object | ... its destination | in vain | wall time |
|---|---|---|---|---|---|
| conflict / cooperation | – / 20, stand-off | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 584 s |
| conflict / conflict | 34 / 24 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 835 s |
| conflict / competition | 43 / 50 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 1348 s |
| conflict / coercion | – / – | 1.00 / 0.99 | 1.00 / 0.01 | 0+0 / 0+0 | 1333 s |
| conflict / exchange | 36 / 45 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 622 s |
| conflict / none | 36 / 45 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 578 s |
| conflict / hinder | – / no goal | 1.00 / 1.00 | 0.33 / 0.99 | 0+0 / 0+0 | 960 s |
