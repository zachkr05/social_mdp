# Which object the partner wants, inferred or known: two_rooms scene, w = 1 / 1, cooperation selective, every robot at levels 2 / 2, swaps on a place, taking from hands, Eq. 7's tau 0.25, a move with the partner's object costs 1 x a move, a hinderer without a goal of its own, inertia: no undoing the last action unless it is worth 1 x a move's cost more than the next best, two posteriors by the partner's hands (its object while it holds nothing, its destination while it holds an object)

Transition at which agent_1 / agent_2 finished its own goal (– = never; no goal: a hinderer without a goal of its own); with the objects inferred, each robot's final belief that its partner wants the partner's own object and that it goes to its true destination (– : a level-0 robot holds no belief); acts for the partner that did not count (set-downs / hand-overs, per robot).

| pairing | inferred | belief: partner's object | ... its destination | in vain | wall time |
|---|---|---|---|---|---|
| conflict / cooperation | – / 20, stand-off | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 618 s |
| conflict / conflict | 18 / 30 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 602 s |
| conflict / competition | 17 / 24 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 490 s |
| conflict / coercion | 17 / 28 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 566 s |
| conflict / exchange | 15 / 22 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 382 s |
| conflict / none | 15 / 22 | 1.00 / 1.00 | 1.00 / 1.00 | 0+0 / 0+0 | 343 s |
| conflict / hinder | 26 / no goal | 1.00 / 1.00 | 0.14 / 1.00 | 0+0 / 0+0 | 708 s |
