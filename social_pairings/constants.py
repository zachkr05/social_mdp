"""Index conventions shared by every module: actions, object statuses, movement outcomes."""
from __future__ import annotations

import numpy as np

# ---- actions ---------------------------------------------------------------------------------
# Four moves, "stay" and five hand actions (since 2026-10-16, in place of the single "use"):
# pick up what lies on its cell, place what it holds there (put it back on its place, put it
# down on a hand-off cell, set it down for the partner on a candidate destination), take what
# the partner next to it holds, give what it holds to the partner next to it, swap what it
# holds for what lies on the object's place it stands on.  A joint action is a0 * 10 + a1.
MOVES = np.array(((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)), dtype=np.int64)   # (5, 2) dx, dy
WAIT = 4
PICK, PLACE, TAKE, GIVE, SWAP = 5, 6, 7, 8, 9
HAND_ACTIONS = (PICK, PLACE, TAKE, GIVE, SWAP)
N_ACTIONS = 10
ACTION_NAMES = ("+y", "+x", "-y", "-x", "stay", "pick", "place", "take", "give", "swap")
MOVE_OF = np.array([0, 1, 2, 3, 4, 4, 4, 4, 4, 4], dtype=np.int64)   # action -> its movement
# JOINT[i, a_own, a_partner]: the joint action when robot i chooses a_own
JOINT = np.array([[[a * N_ACTIONS + b for b in range(N_ACTIONS)] for a in range(N_ACTIONS)],
                  [[b * N_ACTIONS + a for b in range(N_ACTIONS)] for a in range(N_ACTIONS)]],
                 dtype=np.int64)

# ---- objects ---------------------------------------------------------------------------------
HOME, HELD0, HELD1, GONE = 0, 1, 2, 3        # at its place / in R0's hands / in R1's / set down
OBJECT_STATUS = ("at its place", "held by R0", "held by R1", "set down")
# In a scene with hand-off cells (Scene.handoff_cells) a status 4 + h says that the object
# lies on hand-off cell h: put down there by one robot, for the other to pick up.
NO_WAY = 10 ** 6                             # remaining distance while its object is out of reach

# ---- how a robot came to stand on its cell this transition, for the hands rule ----------------
# ARRIVED / STAYED (it waited) / BLOCKED (its move failed), or the hand action it chose:
# HOW_HAND + (action - PICK).  N_HOW codes in all.
ARRIVED, STAYED, BLOCKED = 0, 1, 2
HOW_HAND = 3
HOW_PICK, HOW_PLACE, HOW_TAKE, HOW_GIVE, HOW_SWAP = 3, 4, 5, 6, 7
N_HOW = 8

# ---- task stage of a robot, for ledgers and logs ---------------------------------------------
TO_OBJECT, CARRYING, DONE = 0, 1, 2
STAGE_NAMES = ("to_object", "to_goal", "done")

# ---- solver ----------------------------------------------------------------------------------
CHUNK = 2048                 # joint states per parallel work item
WORK_BYTES = 6 * 8 + 1       # per state and solved model: Krylov vectors (f64) + policy (i8)
TIE_REL = 1e-9               # relative tolerance for treating two Q values as tied
STALL_WARNING = 25           # transitions without progress before a warning is logged
