"""The rule for the hands: who may pick up, hand over, swap and set down what.

Since 2026-10-16 the hands have five actions -- pick, place, take, give, swap -- where they
had one, "use", that did the first of these that applied.

Part 1 walks through the cases that decide whether an object can be held against its owner,
and through hand-off cells and rooms: an object put down for its owner to pick up, robots
that stay in their own rooms; and the two options of 2026-10-12, a swap on an object's
place and taking an object out of the partner's hands.
Part 2 compares every transition table of the world, entry by entry, with an independent
statement of the rule (``reference.Ref``).
Part 3 checks on every transition that an object completes its owner's goal only, on its
owner's destination only.
"""
import unittest

import numpy as np

from social_pairings.constants import (GIVE, GONE, HAND_ACTIONS, HELD0, HELD1, HOME,
                                       N_ACTIONS, PICK, PLACE, SWAP, TAKE, WAIT)
from social_pairings.grid import GridWorld
from social_pairings.rollout import build_world
from social_pairings.world import ObjectWorld

from .reference import OPEN_PASS, PASS, Brute, make_world
from .told import Config

LYING = 4                                    # status of an object on the first hand-off cell

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3           # +y, +x, -y, -x

# R0 starts next to R1's object (0, 0); its own object lies at (2, 2) and belongs on (0, 2).
# R1 starts next to R0's object (2, 2); its own object lies at (0, 0) and belongs on (2, 0).
TINY = dict(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)), pickups=((2, 2), (0, 0)),
            destinations=((0, 2), (2, 0)), candidate_cells=((1, 0),))


def config(**options) -> Config:
    return Config(num_threads=2, **options)


class Case(unittest.TestCase):
    """Helpers to put the world in a state and read the objects off it."""

    options: dict = {}
    scene: dict = TINY

    @classmethod
    def setUpClass(cls):
        cls.cfg = config(**cls.options)
        cls.w, cls.goal_ids = make_world(cls.scene, cls.cfg)

    def state(self, cells, status, done=(0, 0)):
        w = self.w
        return (int(w.cell_id[cells[0]]), int(w.cell_id[cells[1]]), w.block_of(status, done))

    def after(self, state, *joint_actions):
        s = self.w.sid(state)
        for act in joint_actions:
            s = self.w.step(s, act)
        return self.w.unsid(s)

    def objects(self, state) -> tuple:
        return tuple(int(v) for v in self.w.status[state[2]])

    def done(self, state) -> tuple:
        return tuple(int(v) for v in self.w.done[state[2]])

    def may(self, state, robot: int, social: bool, action: int) -> bool:
        """May the robot choose this hand action here (ObjectWorld.hand_ok)?"""
        return bool((self.hands(state, robot, social) >> (action - PICK)) & 1)

    def hands(self, state, robot: int, social: bool) -> int:
        """The bits of the hand actions the robot may choose here (0: none)."""
        w = self.w
        pair = state[0] * w.N + state[1]
        return int(w.hand_ok(social)[robot, state[2], w.uctx[robot, pair]])


# ============================================================================================
# 1. The cases that matter for holding an object against its owner
# ============================================================================================

class OwnObjectTests(Case):
    def test_own_object_is_taken_on_arrival_and_set_down_on_the_destination(self):
        s = self.after(self.w.initial, (UP, WAIT), (RIGHT, WAIT), (RIGHT, WAIT))
        self.assertEqual(self.objects(s), (HELD0, HOME))         # arrived on (2, 2)
        s = self.after(s, (LEFT, WAIT), (LEFT, WAIT))
        self.assertEqual(self.objects(s), (GONE, HOME))          # arrived on (0, 2)
        self.assertEqual(self.done(s), (1, 0))

    def test_a_partners_object_is_not_taken_by_walking_onto_it(self):
        s = self.after(self.w.initial, (DOWN, WAIT))             # R0 onto (0, 0)
        self.assertEqual(self.objects(s), (HOME, HOME))
        self.assertTrue(self.may(s, 0, social=False, action=PICK))
        s = self.after(s, (PICK, WAIT))
        self.assertEqual(self.objects(s), (HOME, HELD0))         # ... but by picking it up

    def test_an_object_is_put_back_only_on_its_own_place(self):
        held = self.state(((0, 0), (2, 1)), (HOME, HELD0))
        self.assertEqual(self.objects(self.after(held, (PLACE, WAIT))), (HOME, HOME))
        elsewhere = self.state(((0, 1), (2, 1)), (HOME, HELD0))
        self.assertEqual(self.hands(elsewhere, 0, social=False), 0)
        self.assertEqual(self.objects(self.after(elsewhere, (PLACE, WAIT))), (HOME, HELD0))

    def test_holding_the_partners_object_leaves_the_partner_nothing_to_do(self):
        """held_reward="none": the goal reward of a robot whose object is in its partner's
        hands is 0 wherever it stands -- what a hinderer is paid for."""
        held = self.state(((0, 0), (1, 1)), (HOME, HELD0))
        free = self.state(((0, 0), (1, 1)), (HOME, HOME))
        self.assertEqual(self.w.rate(held, 1, self.goal_ids[1]), 0.0)
        self.assertGreater(self.w.rate(free, 1, self.goal_ids[1]), 0.0)
        self.assertGreater(self.w.rate(held, 0, self.goal_ids[0]), 0.0)   # its own goal is open


class NobodyLetsGoTests(Case):
    """Each robot holds the other's object, side by side on cells that are no candidates."""

    def hostages(self):
        return self.state(((1, 1), (2, 1)), (HELD1, HELD0))

    def test_one_robot_asking_changes_nothing(self):
        s = self.hostages()
        self.assertEqual(self.objects(self.after(s, (WAIT, GIVE))), (HELD1, HELD0))
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HELD1, HELD0))
        self.assertEqual(self.objects(self.after(s, *[(WAIT, GIVE)] * 20)), (HELD1, HELD0))
        for other in (PICK, PLACE, TAKE, SWAP):          # it gives, the partner does not
            self.assertEqual(self.objects(self.after(s, (GIVE, other))), (HELD1, HELD0))

    def test_both_asking_in_the_same_transition_swaps(self):
        s = self.after(self.hostages(), (GIVE, GIVE))
        self.assertEqual(self.objects(s), (HELD0, HELD1))

    def test_asking_is_a_legal_action_for_a_social_robot_only(self):
        s = self.hostages()
        for robot in (0, 1):
            self.assertEqual(self.hands(s, robot, social=True), 1 << (GIVE - PICK))
            self.assertEqual(self.hands(s, robot, social=False), 0)

    def test_free_hands_receive(self):
        s = self.state(((1, 1), (2, 1)), (HOME, HELD0))          # R0 holds R1's object
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HOME, HELD1))
        self.assertFalse(self.may(s, 0, social=False, action=GIVE))
        # a pick-up where nothing lies leaves the hands free: they receive
        self.assertEqual(self.objects(self.after(s, (GIVE, PICK))), (HOME, HELD1))
        # ... but not hands that pick something up in the same transition
        s = self.state(((2, 1), (2, 2)), (HOME, HELD0))          # R1 on O0's place
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HOME, HELD1))
        self.assertEqual(self.objects(self.after(s, (GIVE, PICK))), (HELD1, HELD0))

    def test_nothing_is_handed_to_a_robot_it_does_not_belong_to(self):
        """R0 holding its OWN object next to R1 cannot give it away."""
        s = self.state(((1, 1), (2, 1)), (HELD0, HOME))
        self.assertEqual(self.hands(s, 0, social=True), 0)
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HELD0, HOME))


class ForcedSwapTests(Case):
    """Config.forced_swap=True, the rule under which nothing can be hindered."""

    options = dict(forced_swap=True)

    def test_one_give_takes_the_own_object_back(self):
        s = self.state(((1, 1), (2, 1)), (HELD1, HELD0))
        # R1 offers R0 its object: R0 has to take it and to give up R1's
        self.assertEqual(self.objects(self.after(s, (WAIT, GIVE))), (HELD0, HELD1))
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HELD0, HELD1))


class SwapOnPlaceTests(Case):
    """``Config.swap_on_place``: full hands on an object's place swap what they hold for
    what lies there.  R0 holds R1's object on the place of its own (2, 2), R1 next to it."""

    options = dict(swap_on_place=True)
    ON_O0 = 4                    # status of an object left on O0's place (no hand-off cells)

    def holding_on_own_place(self):
        return self.state(((2, 2), (2, 1)), (HOME, HELD0))

    def test_swapping_and_handing_over_are_two_actions(self):
        """Until 2026-10-16 "use" swapped here, and nothing was handed over."""
        s = self.after(self.holding_on_own_place(), (SWAP, WAIT))
        self.assertEqual(self.objects(s), (HELD0, self.ON_O0))
        s = self.after(self.holding_on_own_place(), (GIVE, WAIT))
        self.assertEqual(self.objects(s), (HOME, HELD1))
        # with the object's place taken, the held object cannot be placed there
        self.assertFalse(self.may(self.holding_on_own_place(), 0, social=True, action=PLACE))
        s = self.after(self.holding_on_own_place(), (PLACE, WAIT))
        self.assertEqual(self.objects(s), (HOME, HELD0))

    def test_without_the_option_the_object_is_handed_over(self):
        w, _ = make_world(TINY, config())
        s = (int(w.cell_id[(2, 2)]), int(w.cell_id[(2, 1)]), w.block_of((HOME, HELD0), (0, 0)))
        s = w.unsid(w.step(w.sid(s), (GIVE, WAIT)))
        self.assertEqual(tuple(int(v) for v in w.status[s[2]]), (HOME, HELD1))
        s = (int(w.cell_id[(2, 2)]), int(w.cell_id[(2, 1)]), w.block_of((HOME, HELD0), (0, 0)))
        s = w.unsid(w.step(w.sid(s), (SWAP, WAIT)))                  # no swap without it
        self.assertEqual(tuple(int(v) for v in w.status[s[2]]), (HOME, HELD0))

    def test_its_owner_takes_it_from_there_on_arrival(self):
        s = self.after(self.holding_on_own_place(), (SWAP, WAIT), (LEFT, WAIT), (WAIT, UP))
        self.assertEqual(self.objects(s), (HELD0, HELD1))

    def test_swapping_back_puts_an_object_back_on_its_own_place(self):
        s = self.state(((2, 2), (1, 1)), (HELD0, self.ON_O0))
        self.assertEqual(self.objects(self.after(s, (SWAP, WAIT))), (HOME, HELD0))

    def test_every_robot_may_swap(self):
        self.assertTrue(self.may(self.holding_on_own_place(), 0, social=False, action=SWAP))

    def test_nothing_is_swapped_off_a_cell_where_nothing_lies(self):
        s = self.state(((2, 2), (0, 1)), (HELD0, HELD1))       # R0 on its own empty place
        self.assertFalse(self.may(s, 0, social=False, action=SWAP))
        self.assertEqual(self.objects(self.after(s, (SWAP, WAIT))), (HELD0, HELD1))
        self.assertEqual(self.objects(self.after(s, (PLACE, WAIT))), (HOME, HELD1))  # put back

    def test_one_object_to_a_place_and_none_on_its_own_by_a_swap(self):
        w = self.w
        self.assertEqual(w.B, 29)                # 20 without swaps
        for b in range(w.B - 1):
            on = [k if v == HOME else int(v) - 4 if v >= 4 else -1
                  for k, v in enumerate(w.status[b])]
            self.assertEqual(len([p for p in on if p >= 0]), len({p for p in on if p >= 0}))
            self.assertFalse(any(v == 4 + k for k, v in enumerate(w.status[b])))


class StealTests(Case):
    """``Config.steal``: free hands take what the partner next to them holds."""

    options = dict(steal=True)

    def test_free_hands_take_the_partners_object(self):
        s = self.state(((1, 1), (2, 1)), (HELD1, HOME))          # R1 holds R0's object
        self.assertEqual(self.objects(self.after(s, (TAKE, WAIT))), (HELD0, HOME))

    def test_whatever_it_is(self):
        s = self.state(((1, 1), (2, 1)), (HOME, HELD1))          # R1 holds its own
        self.assertEqual(self.objects(self.after(s, (TAKE, WAIT))), (HOME, HELD0))

    def test_without_the_option_nobody_lets_go(self):
        w, _ = make_world(TINY, config())
        s = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of((HELD1, HOME), (0, 0)))
        s = w.unsid(w.step(w.sid(s), (TAKE, WAIT)))
        self.assertEqual(tuple(int(v) for v in w.status[s[2]]), (HELD1, HOME))

    def test_the_holders_own_act_comes_first(self):
        s = self.state(((1, 2), (2, 2)), (HELD1, HOME))          # R1 on O0's place
        self.assertEqual(self.objects(self.after(s, (TAKE, PLACE))), (HOME, HOME))  # put back
        self.assertEqual(self.objects(self.after(s, (TAKE, GIVE))), (HELD0, HOME))  # handed

    def test_only_next_to_the_partner_after_the_moves(self):
        s = self.state(((1, 1), (2, 1)), (HELD1, HOME))
        self.assertEqual(self.objects(self.after(s, (TAKE, DOWN))), (HELD1, HOME))

    def test_full_hands_take_nothing(self):
        s = self.state(((1, 1), (2, 1)), (HELD1, HELD0))
        self.assertFalse(self.may(s, 0, social=True, action=TAKE))
        self.assertEqual(self.objects(self.after(s, (TAKE, WAIT))), (HELD1, HELD0))

    def test_every_robot_may_take(self):
        s = self.state(((1, 1), (2, 1)), (HELD1, HOME))
        self.assertEqual(self.hands(s, 0, social=False), 1 << (TAKE - PICK))
        self.assertEqual(self.hands(self.state(((0, 1), (2, 1)), (HELD1, HOME)), 0,
                                    social=False), 0)            # not next to it


class SetDownForPartnerTests(Case):
    def test_it_counts_on_the_partners_destination(self):
        s = self.state(((2, 0), (1, 2)), (HOME, HELD0))          # R0 on R1's destination
        nxt = self.after(s, (PLACE, WAIT))
        self.assertEqual(self.objects(nxt), (HOME, GONE))
        self.assertEqual(self.done(nxt), (0, 1))

    def test_it_fails_on_another_candidate_and_the_robot_learns_it(self):
        w = self.w
        s = self.state(((1, 0), (1, 2)), (HOME, HELD0))          # a candidate, not the one
        self.assertTrue(self.may(s, 0, social=True, action=PLACE))
        self.assertFalse(self.may(s, 0, social=False, action=PLACE))
        nxt = self.after(s, (PLACE, WAIT))
        self.assertEqual(self.objects(nxt), (HOME, HELD0))
        status, done, failed = w.real_step((HOME, HELD0), (0, 0), s, nxt, (PLACE, WAIT))
        self.assertEqual((status, done, failed), ((HOME, HELD0), (0, 0), (True, False)))

    def test_a_partners_object_cannot_be_set_down_on_the_holders_own_destination(self):
        """R1 stands on its own destination (2, 0) with R0's object in its hands and places
        it: nothing is set down, and nothing is done.  The object stays in its hands."""
        w = self.w
        s = self.state(((0, 2), (2, 0)), (HELD1, HOME))
        self.assertEqual(w.cell(s[1]), w.cell(w.true_dest[1]))
        nxt = self.after(s, (WAIT, PLACE))
        self.assertEqual((self.objects(nxt), self.done(nxt)), ((HELD1, HOME), (0, 0)))
        status, done, failed = w.real_step((HELD1, HOME), (0, 0), s, nxt, (WAIT, PLACE))
        self.assertEqual((status, done, failed), ((HELD1, HOME), (0, 0), (False, True)))
        # ... and walking onto its own destination with it does not set it down either
        s = self.state(((0, 2), (2, 1)), (HELD1, HOME))
        nxt = self.after(s, (WAIT, DOWN))
        self.assertEqual(w.cell(nxt[1]), w.cell(w.true_dest[1]))
        self.assertEqual((self.objects(nxt), self.done(nxt)), ((HELD1, HOME), (0, 0)))

    def test_a_hypothetical_model_can_let_it_count_anywhere(self):
        """The solver mixes two tables by the belief that the cell is the destination."""
        w = self.w
        s = self.state(((1, 0), (1, 2)), (HOME, HELD0))
        pair = s[0] * w.N + s[1]
        event = w.pev[pair, PLACE * N_ACTIONS + WAIT]
        fails = w.model_tab(0, self.goal_ids[0])[s[2], event]
        counts = w.model_tab(0, self.goal_ids[0], True)[s[2], event]
        self.assertEqual(tuple(w.status[fails]), (HOME, HELD0))
        self.assertEqual(tuple(w.status[counts]), (HOME, GONE))


class HandOffCellTests(Case):
    """``PASS`` without rooms: the doorway (2, 0) is a hand-off cell.  R0's object lies at
    (4, 1), R1's at (0, 1)."""

    scene = OPEN_PASS
    door = (2, 0)

    def test_the_partners_object_is_put_down_there_and_lies_there(self):
        s = self.state((self.door, (4, 0)), (HOME, HELD0))       # R0 holds R1's object
        self.assertTrue(self.may(s, 0, social=True, action=PLACE))
        self.assertFalse(self.may(s, 0, social=False, action=PLACE))   # an act for the partner
        nxt = self.after(s, (PLACE, WAIT))
        self.assertEqual(self.objects(nxt), (HOME, LYING))
        self.assertEqual(self.done(nxt), (0, 0))
        self.assertEqual(self.w.object_cells(self.objects(nxt), nxt), ((4, 1), self.door))

    def test_nowhere_else_and_not_its_own_object(self):
        """Only on the hand-off cell; and a robot that knows which object its partner wants
        may put down that one only.  Putting down itself works with any object (what lies
        there anybody may pick up): since 2026-10-09, for a robot that does not know which
        object its partner wants (``partner_object="inferred"``, test_objects.py)."""
        elsewhere = self.state(((1, 0), (4, 0)), (HOME, HELD0))
        self.assertEqual(self.hands(elsewhere, 0, social=True), 0)
        self.assertEqual(self.objects(self.after(elsewhere, (PLACE, WAIT))), (HOME, HELD0))
        own = self.state((self.door, (4, 0)), (HELD0, HOME))     # R0 holds its own object
        self.assertEqual(self.hands(own, 0, social=True), 0)
        self.assertEqual(self.objects(self.after(own, (PLACE, WAIT))), (LYING, HOME))

    def test_its_owner_takes_it_on_arrival(self):
        s = self.state(((1, 0), (3, 0)), (HOME, LYING))
        nxt = self.after(s, (WAIT, LEFT))                        # R1 steps into the doorway
        self.assertEqual(self.objects(nxt), (HOME, HELD1))
        # ... and its partner does not: it would have to pick it up
        nxt = self.after(s, (RIGHT, WAIT))
        self.assertEqual(self.objects(nxt), (HOME, LYING))
        self.assertEqual(self.objects(self.after(nxt, (PICK, WAIT))), (HOME, HELD0))

    def test_full_hands_do_not_take_it(self):
        s = self.state(((0, 0), (3, 0)), (HELD1, LYING))         # R1 holds R0's object
        nxt = self.after(s, (WAIT, LEFT))
        self.assertEqual(self.objects(nxt), (HELD1, LYING))
        nxt = self.after(nxt, (WAIT, PLACE))                     # it puts R0's object down
        self.assertEqual(self.objects(nxt), (LYING, LYING))
        nxt = self.after(nxt, (WAIT, PICK))                      # and picks up ITS OWN first
        self.assertEqual(self.objects(nxt), (LYING, HELD1))
        nxt = self.after(nxt, (WAIT, RIGHT), (RIGHT, WAIT), (RIGHT, WAIT))
        self.assertEqual(self.objects(nxt), (HELD0, HELD1))      # R0 arrives and takes its own

    def test_its_reward_follows_where_it_lies(self):
        """R1 next to the doorway: with its object on the doorway its way is one step to it
        and two on to (4, 0); at its place, (0, 1), it is four steps and five back."""
        w, g = self.w, self.goal_ids[1]
        lying = self.state(((1, 0), (3, 0)), (HOME, LYING))
        home = self.state(((1, 0), (3, 0)), (HOME, HOME))
        self.assertEqual((w.dist_left(lying, 1, g), w.dist_left(home, 1, g)), (3, 9))
        self.assertGreater(w.rate(lying, 1, g), w.rate(home, 1, g))

    def test_on_a_hand_off_cell_an_object_is_put_down_or_handed_over(self):
        """Two actions where "use" (until 2026-10-16) only put the object down."""
        s = self.state((self.door, (3, 0)), (HOME, HELD0))       # R0 in the doorway, R1 next
        self.assertEqual(self.hands(s, 0, social=True),
                         (1 << (PLACE - PICK)) | (1 << (GIVE - PICK)))
        self.assertEqual(self.objects(self.after(s, (PLACE, WAIT))), (HOME, LYING))
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HOME, HELD1))
        # ... and the robot NEXT to the doorway hands over to the one standing in it
        s = self.state((self.door, (3, 0)), (HELD1, HOME))       # R1 holds R0's object
        self.assertTrue(self.may(s, 1, social=True, action=GIVE))
        self.assertEqual(self.hands(s, 1, social=False), 0)
        self.assertEqual(self.objects(self.after(s, (WAIT, GIVE))), (HELD0, HOME))

    def test_two_robots_at_the_doorway(self):
        s = self.state((self.door, (3, 0)), (HELD1, HELD0))
        # the robot in the doorway puts its partner's object down: a hand-over to it fails
        self.assertEqual(self.objects(self.after(s, (PLACE, GIVE))), (HELD1, LYING))
        self.assertEqual(self.objects(self.after(s, (PLACE, WAIT))), (HELD1, LYING))
        self.assertEqual(self.objects(self.after(s, (WAIT, GIVE))), (HELD1, HELD0))
        # one transition later its hands are free, and it is handed its own object
        self.assertEqual(self.objects(self.after(s, (PLACE, WAIT), (WAIT, GIVE))),
                         (HELD0, LYING))
        # both give: they trade, in the doorway as on every map
        self.assertEqual(self.objects(self.after(s, (GIVE, GIVE))), (HELD0, HELD1))
        apart = self.state(((1, 0), (0, 0)), (HELD1, HELD0))
        self.assertEqual(self.objects(self.after(apart, (GIVE, GIVE))), (HELD0, HELD1))

    def test_a_robot_without_social_goal_does_nothing_for_its_partner(self):
        """Wherever a hand action is legal for it while it holds its partner's object, it is
        "place" on the object's place, which puts it back: it does not end in its partner's
        hands, on a hand-off cell or on a destination, whatever the partner does in the same
        transition."""
        w = self.w
        allowed = w.hand_ok(False)
        checked = 0
        for b in range(w.B - 1):
            for i in (0, 1):
                if w.hold[b, i] != 1 - i:
                    continue
                for pair in np.flatnonzero(w.pair_ok).tolist():
                    bits = int(allowed[i, b, w.uctx[i, pair]])
                    if not bits:
                        continue
                    self.assertEqual(bits, 1 << (PLACE - PICK))
                    self.assertEqual(pair // w.N if i == 0 else pair % w.N, w.home[1 - i])
                    for partner in range(N_ACTIONS):
                        act = (PLACE, partner) if i == 0 else (partner, PLACE)
                        nxt = w.unsid(w.step(b * w.NN + pair, act))
                        self.assertEqual(int(w.status[nxt[2], 1 - i]), HOME, (b, pair, act))
                        checked += 1
        self.assertGreater(checked, 100)

    def test_a_hand_off_cell_is_no_place_and_no_destination(self):
        for clash in (dict(handoff_cells=((0, 1),)), dict(handoff_cells=((4, 0),))):
            with self.assertRaises(ValueError):
                make_world(dict(OPEN_PASS, **clash), config())


class NoHandOverTests(Case):
    """``Config.hand_over=False``: nothing goes from hand to hand."""

    scene = OPEN_PASS
    options = dict(hand_over=False)

    def test_next_to_the_partner_the_object_is_put_down_not_handed_over(self):
        s = self.state(((2, 0), (3, 0)), (HOME, HELD0))
        self.assertEqual(self.hands(s, 0, social=True), 1 << (PLACE - PICK))
        self.assertEqual(self.objects(self.after(s, (PLACE, WAIT))), (HOME, LYING))
        self.assertEqual(self.objects(self.after(s, (GIVE, WAIT))), (HOME, HELD0))

    def test_both_giving_do_not_trade(self):
        s = self.state(((2, 0), (3, 0)), (HELD1, HELD0))
        self.assertEqual(self.objects(self.after(s, (GIVE, GIVE))), (HELD1, HELD0))
        self.assertEqual(self.objects(self.after(s, (PLACE, GIVE))), (HELD1, LYING))
        self.assertEqual(self.hands(s, 1, social=True), 0)       # R1: nothing its hands can do

    def test_away_from_a_hand_off_cell_there_is_no_hand_action(self):
        s = self.state(((1, 0), (0, 0)), (HOME, HELD0))          # side by side, no hand-off cell
        self.assertEqual(self.hands(s, 0, social=True), 0)
        for a in HAND_ACTIONS:
            self.assertEqual(self.objects(self.after(s, (a, WAIT))), (HOME, HELD0))


class RoomTests(Case):
    """``PASS``: each robot stays in its own room; the doorway (2, 0) belongs to both."""

    scene = PASS

    def test_a_robot_does_not_leave_its_room(self):
        w = self.w
        door, beyond = int(w.cell_id[(2, 0)]), int(w.cell_id[(3, 0)])
        self.assertTrue(w.valid_by[1, door, RIGHT] and not w.valid_by[0, door, RIGHT])
        self.assertTrue(w.valid_by[0, door, LEFT] and not w.valid_by[1, door, LEFT])
        s = self.state(((2, 0), (4, 0)), (HOME, HOME))
        self.assertEqual(self.after(s, (RIGHT, WAIT)), s)        # no move at all
        self.assertFalse(w.pair_ok[beyond * w.N + int(w.cell_id[(4, 0)])])

    def test_every_pair_of_moves_against_the_rule_written_out(self):
        for scene in (PASS, OPEN_PASS):
            w, _ = make_world(scene, self.cfg)
            ref = Brute(w, self.cfg)
            for pair in ref.pairs:
                x = divmod(pair, w.N)
                for m0 in range(5):
                    for m1 in range(5):
                        self.assertEqual(divmod(int(w.grid.pair_moves[pair, m0 * 5 + m1]), w.N),
                                         ref.pair_move(x, (m0, m1)), (x, m0, m1))

    def test_an_object_in_the_other_room_is_out_of_reach(self):
        w = self.w
        for i in (0, 1):
            self.assertFalse(w.can_finish_alone(i, self.goal_ids[i]))
            self.assertEqual(w.rate(w.initial, i, self.goal_ids[i]), 0.0)
        self.assertEqual(w.solo_preflight(self.goal_ids), (None, None))
        # on the doorway it is not
        s = self.state(((1, 0), (3, 0)), (HOME, LYING))
        self.assertGreater(w.rate(s, 1, self.goal_ids[1]), 0.0)
        self.assertEqual(w.rate(s, 0, self.goal_ids[0]), 0.0)

    def test_a_robot_has_to_start_in_its_room(self):
        from social_pairings import SolverFailure
        with self.assertRaises(SolverFailure):
            make_world(dict(PASS, starts=((3, 0), (3, 1))), self.cfg)

    def test_without_rooms_every_robot_has_the_moves_of_the_grid(self):
        w, _ = make_world(OPEN_PASS, self.cfg)
        self.assertFalse(w.has_rooms)
        for i in (0, 1):
            self.assertTrue((w.valid_by[i] == w.valid).all() and (w.nb_by[i] == w.nb).all())
            self.assertTrue((w.d_goal_by[i] == w.d_goal).all())
            self.assertTrue((w.d_obj_by[i] == w.d_obj).all())


class SpareObjectTests(unittest.TestCase):
    """A third object on (1, 2) that completes nobody's goal."""

    scene = dict(TINY, objects=((1, 2),))

    def world(self, **options):
        cfg = config(**options)
        world, goal_ids, _, scenery = build_world(
            self.scene["grid"], self.scene["starts"], self.scene["pickups"],
            self.scene["destinations"], self.scene["candidate_cells"], self.scene["objects"], cfg)
        return world, goal_ids, scenery

    def test_by_default_a_spare_object_is_scenery(self):
        w, _, scenery = self.world()
        self.assertEqual((w.M, scenery), (2, ((1, 2),)))
        on_it = (int(w.cell_id[(1, 2)]), int(w.cell_id[(2, 1)]), w.block0)
        self.assertEqual(w.hand_ok(True)[0, w.block0, w.uctx[0, on_it[0] * w.N + on_it[1]]], 0)
        self.assertEqual(w.unsid(w.step(w.sid(on_it), (PICK, WAIT))), on_it)

    def test_handle_spares_lets_it_be_picked_up_but_not_given_away(self):
        w, _, scenery = self.world(handle_spares=True)
        self.assertEqual((w.M, scenery), (3, ()))
        on_it = (int(w.cell_id[(1, 2)]), int(w.cell_id[(2, 2)]), w.block0)
        nxt = w.unsid(w.step(w.sid(on_it), (PICK, WAIT)))
        self.assertEqual(tuple(w.status[nxt[2]]), (HOME, HOME, HELD0))
        # next to its partner, holding the spare: no hand-over, not even for a social robot
        beside = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), nxt[2])
        self.assertEqual(w.hand_ok(True)[0, nxt[2], w.uctx[0, beside[0] * w.N + beside[1]]], 0)
        self.assertEqual(w.unsid(w.step(w.sid(beside), (GIVE, WAIT))), beside)

    def test_scenery_is_the_same_world_as_objects_nobody_may_handle(self):
        """Leaving the spare objects out of the model changes nothing: with the spare in
        the model but not to be handled, every transition from every reachable
        configuration moves the two own objects exactly as in the two-object world."""
        cfg = config()
        small, goal_ids, _ = self.world()
        candidates = tuple(dict.fromkeys((*self.scene["candidate_cells"],
                                          *self.scene["destinations"])))
        grid = GridWorld(self.scene["grid"], self.scene["starts"], self.scene["pickups"],
                         candidates, cfg, goal_ids)
        big = ObjectWorld(grid, cfg, self.scene["pickups"] + self.scene["objects"])
        self.assertEqual((small.M, big.M), (2, 3))
        pairs = np.flatnonzero(small.pair_ok)
        tables = [(small.true_tab, big.true_tab)]
        for o in (0, 1):
            for success in (False, True):
                tables.append((small.model_tab(o, goal_ids[o], success),
                               big.model_tab(o, goal_ids[o], success)))
            tables.append((small.level0_tab(o, goal_ids[o]), big.level0_tab(o, goal_ids[o])))
        checked = 0
        for b in range(small.B - 1):
            status, done = tuple(small.status[b]), tuple(small.done[b])
            b_big = big.block_of(status + (HOME,), done)
            for tab_small, tab_big in tables:
                nxt_small = tab_small[b][small.pev[pairs]]                     # (pairs, 100)
                nxt_big = tab_big[b_big][big.pev[pairs]]
                over = nxt_small == small.T          # both goals done: one terminal block
                self.assertTrue((over == (nxt_big == big.T)).all())
                after_small, after_big = nxt_small[~over], nxt_big[~over]
                self.assertTrue((big.status[after_big][:, 2] == HOME).all())
                self.assertTrue((big.status[after_big][:, :2] == small.status[after_small]).all())
                self.assertTrue((big.done[after_big] == small.done[after_small]).all())
                checked += nxt_small.size
            for social in (False, True):
                for i in (0, 1):
                    self.assertTrue((big.hand_ok(social)[i, b_big][big.uctx[i, pairs]]
                                     == small.hand_ok(social)[i, b][small.uctx[i, pairs]]).all())
        self.assertGreater(checked, 300000)


# ============================================================================================
# 2. Every table against the independent statement of the rule
# ============================================================================================

class RuleAgainstReferenceTests(unittest.TestCase):
    def compare(self, scene, cfg, models=((0, None), (1, None))):
        """Every dynamics the solver uses, for every configuration, pair of cells and joint
        action; and which hand actions each state allows.  models: (robot, destination) pairs
        whose model tables are compared (None = the robot's true destination)."""
        w, goal_ids = make_world(scene, cfg)
        ref = Brute(w, cfg)
        pairs = np.array(ref.pairs)
        dest = [int(c) for c in w.dest]
        true = tuple(dest[g] for g in goal_ids)
        cases = [("true", w.true_tab, (true, (true[1], true[0]), (0, 1), (True, True)))]
        for o, k in models:
            k = goal_ids[o] if k is None else k
            goal = tuple(dest[k] if i == o else None for i in (0, 1))
            cases += [
                (f"model R{o} g{k}, set-down fails", w.model_tab(o, k),
                 (goal, tuple(None if i == o else dest[k] for i in (0, 1)), (0, 1), (True, True))),
                (f"model R{o} g{k}, set-down counts", w.model_tab(o, k, True),
                 (goal, tuple("any" if i == o else dest[k] for i in (0, 1)), (0, 1),
                  (True, True))),
                (f"level 0 R{o} g{k}", w.level0_tab(o, k),
                 (goal, (None, None), tuple(i if i == o else None for i in (0, 1)),
                  tuple(i == o for i in (0, 1))))]
        entries = 0
        for name, table, key in cases:
            expected = ref.nexts(*key)[:-1][:, pairs, :]                   # (B - 1, pairs, 100)
            got = table[:-1][:, w.pev[pairs]]                               # the same, by event
            wrong = np.argwhere(expected != got)
            self.assertEqual(len(wrong), 0, f"{name}: {len(wrong)} transitions differ, first "
                                            f"(block, pair, joint action) {wrong[:1].tolist()}")
            entries += expected.size
        for social in (False, True):
            allowed = w.hand_ok(social)
            for b in range(w.B - 1):
                for pair in ref.pairs:
                    x = divmod(pair, w.N)
                    for i in (0, 1):
                        bits = int(allowed[i, b, w.uctx[i, pair]])
                        for a in HAND_ACTIONS:
                            self.assertEqual(bool((bits >> (a - PICK)) & 1),
                                             ref.may(b, i, x, social, a), (social, b, x, i, a))
        return entries

    def test_default_rule(self):
        self.assertGreater(self.compare(TINY, config(), ((0, None), (1, 0))), 300000)

    def test_forced_swap(self):
        self.compare(TINY, config(forced_swap=True))

    def test_hand_overs_only(self):
        self.compare(TINY, config(set_down_for_partner=False), ((1, None),))

    def test_done_robots_keep_their_hands_still(self):
        self.compare(TINY, config(done_robots_handle=False), ((0, None),))

    def test_spare_object_that_may_be_handled(self):
        scene = dict(grid=np.ones((3, 2)), starts=((0, 0), (2, 0)), pickups=((2, 1), (0, 1)),
                     destinations=((1, 0), (1, 1)), candidate_cells=(), objects=((2, 0),))
        self.compare(scene, config(handle_spares=True), ((0, None),))

    def test_a_hand_off_cell(self):
        self.assertGreater(self.compare(OPEN_PASS, config()), 300000)

    def test_a_hand_off_cell_and_rooms(self):
        self.compare(PASS, config(set_down_for_partner=False), ((0, None), (1, 0)))

    def test_a_hand_off_cell_without_hand_overs(self):
        self.compare(PASS, config(set_down_for_partner=False, hand_over=False))
        self.compare(OPEN_PASS, config(hand_over=False), ((1, None),))

    def test_a_hand_off_cell_under_forced_swap(self):
        self.compare(OPEN_PASS, config(forced_swap=True), ((0, None),))

    def test_swaps_and_steals(self):
        for options in (dict(swap_on_place=True), dict(steal=True),
                        dict(swap_on_place=True, steal=True),
                        dict(swap_on_place=True, steal=True, forced_swap=True),
                        dict(swap_on_place=True, steal=True, hand_over=False),
                        dict(swap_on_place=True, steal=True, done_robots_handle=False)):
            with self.subTest(**options):
                self.compare(TINY, config(**options), ((0, None), (1, 0)))
        self.compare(OPEN_PASS, config(swap_on_place=True, steal=True), ((0, None),))

    # ---- a robot that does not know which object its partner wants ---------------------
    def compare_inferred(self, scene, cfg, models):
        """``partner_object="inferred"``: the tables of a model of robot o with physical goal
        (object q, destination k) -- its partner takes no object offered (set-down fails) or
        any (counts, wherever it is set down), takes none on arriving and picks none first
        -- and the robot's level-0 table with goal object q; every configuration, pair of
        cells and joint action; and which hand actions each state allows (any object may be
        offered)."""
        w, goal_ids = make_world(scene, cfg)
        self.assertTrue(w.infer_object)
        ref = Brute(w, cfg)
        pairs = np.array(ref.pairs)
        dest = [int(c) for c in w.dest]
        true = tuple(dest[g] for g in goal_ids)
        cases = [("true", w.true_tab, (true, (true[1], true[0]), (0, 1), (True, True)))]
        for o, k, q in models:
            goal = tuple(dest[k] if i == o else None for i in (0, 1))
            own = tuple(q if i == o else None for i in (0, 1))
            cases += [
                (f"model R{o} O{q} g{k}, act fails", w.model_tab(o, k, obj=q),
                 (goal, tuple(None if i == o else dest[k] for i in (0, 1)), own, (True, True),
                  own, own)),
                (f"model R{o} O{q} g{k}, act counts", w.model_tab(o, k, True, obj=q),
                 (goal, tuple("any" if i == o else dest[k] for i in (0, 1)), own, (True, True),
                  own, tuple(q if i == o else "any" for i in (0, 1)))),
                (f"level 0 R{o} O{q} g{k}", w.level0_tab(o, k, q),
                 (goal, (None, None), own, tuple(i == o for i in (0, 1)),
                  tuple(q if i == o else i for i in (0, 1)),
                  tuple(q if i == o else i for i in (0, 1))))]
        for name, table, key in cases:
            expected = ref.nexts(*key)[:-1][:, pairs, :]
            got = table[:-1][:, w.pev[pairs]]
            wrong = np.argwhere(expected != got)
            self.assertEqual(len(wrong), 0, f"{name}: {len(wrong)} transitions differ, first "
                                            f"(block, pair, joint action) {wrong[:1].tolist()}")
        for social in (False, True):
            allowed = w.hand_ok(social)
            for b in range(w.B - 1):
                for pair in ref.pairs:
                    x = divmod(pair, w.N)
                    for i in (0, 1):
                        bits = int(allowed[i, b, w.uctx[i, pair]])
                        for a in HAND_ACTIONS:
                            self.assertEqual(bool((bits >> (a - PICK)) & 1),
                                             ref.may(b, i, x, social, a), (social, b, x, i, a))

    def test_inferred_partner_objects(self):
        cfg = config(partner_object="inferred")
        self.compare_inferred(TINY, cfg, ((0, 0, 0), (0, 2, 1), (1, 1, 0), (1, 2, 1)))

    def test_inferred_partner_objects_with_a_spare_one(self):
        scene = dict(grid=np.ones((3, 2)), starts=((0, 0), (2, 0)), pickups=((2, 1), (0, 1)),
                     destinations=((1, 0), (1, 1)), candidate_cells=(), objects=((2, 0),))
        cfg = config(partner_object="inferred")
        w, _ = make_world(scene, cfg)
        self.assertEqual(w.M, 3)                 # the spare object is in the model
        self.compare_inferred(scene, cfg, ((0, 0, 2), (1, 1, 0), (1, 0, 2)))

    def test_inferred_partner_objects_on_a_hand_off_cell(self):
        cfg = config(partner_object="inferred", set_down_for_partner=False)
        self.compare_inferred(PASS, cfg, ((0, 0, 1), (1, 1, 1), (1, 0, 0)))
        self.compare_inferred(OPEN_PASS, config(partner_object="inferred"),
                              ((0, 0, 0), (1, 0, 0)))

    def test_inferred_partner_objects_with_swaps_and_steals(self):
        spare = dict(grid=np.ones((3, 2)), starts=((0, 0), (2, 0)), pickups=((2, 1), (0, 1)),
                     destinations=((1, 0), (1, 1)), candidate_cells=(), objects=((2, 0),))
        cfg = config(partner_object="inferred", swap_on_place=True, steal=True)
        self.compare_inferred(TINY, cfg, ((0, 0, 0), (0, 2, 1), (1, 1, 0), (1, 2, 1)))
        self.compare_inferred(spare, cfg, ((0, 0, 2), (1, 1, 0), (1, 0, 2)))
        self.compare_inferred(OPEN_PASS, cfg, ((0, 0, 0), (1, 0, 0)))


# ============================================================================================
# 3. An object completes its owner's goal, and nobody else's
# ============================================================================================

class OwnObjectRuleTests(unittest.TestCase):
    """Robot i's goal is done by object i lying on robot i's destination, and by nothing
    else.  An object is at its place, in a pair of hands, or on its owner's destination; so
    "red picks up orange's object and places it on red's destination" is not a transition
    of this world, and there is no option that makes it one.

    Checked from every configuration in which exactly the delivered objects' owners are
    done, for every pair of cells and every joint action: on the transition table, and --
    wherever a transition changes the configuration -- on ``real_step``, the rule the
    record of a rollout is written by, which keeps the objects apart to the end."""

    def check(self, w) -> int:
        status, done = np.asarray(w.status), np.asarray(w.done)             # (B, M), (B, 2)
        owned = (status[:, :2] == GONE) == (done == 1)       # delivered <=> its owner is done
        sound = owned.all(axis=1) & (status[:, 2:] != GONE).all(axis=1)     # (B,)
        sound[w.T] = True                # both done: one block for every such configuration
        self.assertTrue(sound[w.initial[2]] and done[w.T].all())
        pairs = np.flatnonzero(w.pair_ok)
        dest = [int(c) for c in w.true_dest]
        checked = delivered = 0
        for b in np.flatnonzero(sound).tolist():
            if b == w.T:
                continue
            after = w.true_tab[b][w.pev[pairs]]                             # (pairs, 100)
            self.assertTrue(sound[after].all(),
                            "a transition delivers an object for a robot that does not own it")
            checked += after.size
            before = tuple(int(v) for v in status[b])
            for p, j in np.argwhere(after != b).tolist():   # something happened to an object
                act = divmod(j, N_ACTIONS)
                state = (*divmod(int(pairs[p]), w.N), b)
                nxt = w.unsid(w.step(w.sid(state), act))
                st, dn, _ = w.real_step(before, tuple(int(v) for v in done[b]), state, nxt, act)
                where = (before, state, act)
                self.assertEqual(w.block_of(st, dn), nxt[2], where)   # one rule, two places
                self.assertEqual([v == GONE for v in st[:2]], [bool(v) for v in dn], where)
                self.assertNotIn(GONE, st[2:], where)                 # a spare is never set down
                self.assertTrue(all(dn[i] >= done[b][i] for i in (0, 1)), where)
                near = int(np.abs(w.cells[nxt[0]] - w.cells[nxt[1]]).sum()) == 1
                for k in (0, 1):
                    if st[k] != GONE or before[k] == GONE:
                        continue
                    delivered += 1
                    holder = {HELD0: 0, HELD1: 1}.get(before[k], k)   # at its place: its owner
                    home = nxt[k] == dest[k]         # its owner stands on its own destination
                    # set down by its owner there; or by its partner on the owner's
                    # destination, or handed to the owner standing there
                    self.assertTrue(home if holder == k else
                                    (nxt[holder] == dest[k] or (near and home)), where)
        self.assertGreater(delivered, 0)
        return checked

    def test_default_rule(self):
        self.assertGreater(self.check(make_world(TINY, config())[0]), 25000)

    def test_there_is_no_switch_for_it(self):
        with self.assertRaises(TypeError):
            Config(goal_objects="any")
        self.assertFalse(hasattr(config(), "goal_objects"))

    def test_the_check_notices_a_world_without_the_rule(self):
        """Two worlds with the rule broken by hand in a single entry of the transition
        table, and the check says so.  (1) R1 stands on its own destination with R0's
        object in its hands and places it, and the object counts for R1: "red places
        orange's object on red's destination".  (2) R0 walks one cell with its own object
        in its hands, away from its destination, and the object is delivered."""
        w, _ = make_world(TINY, config())
        held_by_partner = (int(w.cell_id[(0, 2)]), int(w.cell_id[(2, 0)]),
                           w.block_of([HELD1, HOME], [0, 0]))
        self.assertEqual(w.cell(held_by_partner[1]), w.cell(w.true_dest[1]))
        event = w.pev[held_by_partner[0] * w.N + held_by_partner[1], WAIT * N_ACTIONS + PLACE]
        self.assertEqual(w.true_tab[held_by_partner[2], event], held_by_partner[2])
        w.true_tab[held_by_partner[2], event] = w.block_of([GONE, HOME], [0, 1])
        with self.assertRaisesRegex(AssertionError, "does not own it"):
            self.check(w)

        w, _ = make_world(TINY, config())
        carrying = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]),
                    w.block_of([HELD0, HOME], [0, 0]))
        self.assertNotEqual(w.cell(w.true_dest[0]), (1, 0))
        event = w.pev[carrying[0] * w.N + carrying[1], DOWN * N_ACTIONS + WAIT]
        self.assertEqual(w.true_tab[carrying[2], event], carrying[2])
        w.true_tab[carrying[2], event] = w.block_of([GONE, HOME], [1, 0])
        with self.assertRaises(AssertionError):
            self.check(w)

    def test_under_every_option_of_the_hands_rule(self):
        for options in (dict(forced_swap=True), dict(set_down_for_partner=False),
                        dict(done_robots_handle=False), dict(hand_over=False),
                        dict(swap_on_place=True, steal=True)):
            with self.subTest(**options):
                self.check(make_world(TINY, config(**options))[0])

    def test_with_a_hand_off_cell_and_rooms(self):
        """An object that lies on a hand-off cell completes nothing: it is delivered only
        from a pair of hands, as everywhere."""
        for scene, options in ((OPEN_PASS, {}), (PASS, dict(set_down_for_partner=False)),
                               (PASS, dict(set_down_for_partner=False, hand_over=False))):
            with self.subTest(rooms=scene["rooms"] is not None, **options):
                self.check(make_world(scene, config(**options))[0])

    def test_with_spare_objects_in_play(self):
        """A spare object can be picked up (``handle_spares=True``) and never set down on
        a destination: it completes nobody's goal."""
        scene = dict(TINY, objects=((1, 2),))
        self.assertGreater(self.check(make_world(scene, config(handle_spares=True))[0]), 60000)

    def test_on_the_six_object_scene(self):
        from social_pairings import six_object_scene
        scene = six_object_scene()
        described = dict(grid=scene.grid, starts=scene.starts, pickups=scene.pickups,
                         destinations=scene.destinations,
                         candidate_cells=scene.candidate_cells, objects=scene.objects)
        self.assertGreater(self.check(make_world(described, config())[0]), 300000)


if __name__ == "__main__":
    unittest.main()
