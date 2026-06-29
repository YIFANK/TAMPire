"""Procedurally generate randomized, solvable tabletop scenes with a
natural-language goal and the ground-truth goal predicates.

Difficulty knobs that exercise the planner/verifier:
  - stacks: a block sits on the goal/target block (must be cleared first)
  - clutter: extra distractor blocks
  - reach: occasionally place a distractor out of the arm's reach (the verifier
    enforces reach; a naive planner that grabs the wrong thing will be caught)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..schemas import Predicate, Scene, WorldObject
from ..sim.world import BASE_XY, REACH_RADIUS

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
TABLE = (-0.4, -0.3, 0.4, 0.3)


@dataclass
class EvalTask:
    scene: Scene
    goal: str
    goal_predicates: List[Predicate]
    kind: str           # "in_bowl" | "stack"
    difficulty: str     # "easy" | "trap" | "clutter"


def _reachable_xy(rng: random.Random) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = TABLE
    for _ in range(200):
        x = rng.uniform(xmin + 0.05, xmax - 0.05)
        y = rng.uniform(ymin + 0.05, ymax - 0.05)
        if math.hypot(x - BASE_XY[0], y - BASE_XY[1]) <= REACH_RADIUS - 0.05:
            return round(x, 3), round(y, 3)
    return 0.0, 0.0


def _block(oid: str, color: str, xy: Tuple[float, float], z: float = 0.04) -> WorldObject:
    return WorldObject(id=oid, category="block", color=color,
                       position=(xy[0], xy[1], z), size=(0.04, 0.04, 0.04),
                       affordances=["graspable", "stackable"])


def _spaced_positions(rng: random.Random, n: int, min_d: float = 0.13) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < 2000:
        tries += 1
        p = _reachable_xy(rng)
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_d for q in pts):
            pts.append(p)
    return pts


def generate(seed: int) -> EvalTask:
    rng = random.Random(seed)
    kind = rng.choice(["in_bowl", "stack"])
    difficulty = rng.choices(["easy", "trap", "clutter"], weights=[0.3, 0.45, 0.25])[0]

    n_blocks = {"easy": 2, "trap": 2, "clutter": rng.randint(3, 4)}[difficulty]
    colors = rng.sample(COLORS, n_blocks + (1 if kind == "in_bowl" else 0))

    slots = _spaced_positions(rng, n_blocks + 1)  # +1 for bowl/target area
    objects: List[WorldObject] = []
    preds: List[Predicate] = []

    block_ids = []
    for i in range(n_blocks):
        oid = f"{colors[i]}_block"
        block_ids.append(oid)
        objects.append(_block(oid, colors[i], slots[i]))
        preds.append(Predicate("on", [oid, "table"]))

    # mark all clear initially; we revoke for the stacked one below
    clear_ids = set(block_ids)

    if difficulty == "trap":
        # stack a distractor block on the FIRST block (the eventual goal object)
        target_block = block_ids[0]
        stacker = block_ids[1]
        sx, sy, sz = objects[0].position
        # move stacker on top of target
        objects[1].position = (sx, sy, sz + 0.04)
        preds = [p for p in preds if not (p.name == "on" and p.args[0] == stacker)]
        preds.append(Predicate("on", [stacker, target_block]))
        clear_ids.discard(target_block)

    if kind == "in_bowl":
        bowl_color = colors[-1]
        bowl = WorldObject(id=f"{bowl_color}_bowl", category="bowl", color=bowl_color,
                           position=(slots[n_blocks][0], slots[n_blocks][1], 0.03),
                           size=(0.12, 0.12, 0.05), affordances=["container", "support"])
        objects.append(bowl)
        clear_ids.add(bowl.id)
        goal_obj = block_ids[0]
        goal = f"put the {objects[0].color} block in the {bowl.color} bowl"
        goal_predicates = [Predicate("in", [goal_obj, bowl.id])]
    else:  # stack
        a, b = block_ids[0], block_ids[-1]  # stack a onto b (a may be trapped under stacker)
        # ensure a != b
        if a == b and len(block_ids) > 1:
            b = block_ids[1]
        goal = f"stack the {objects[0].color} block on the {next(o.color for o in objects if o.id == b)} block"
        goal_predicates = [Predicate("on", [a, b])]

    for cid in clear_ids:
        preds.append(Predicate("clear", [cid]))
    for oid in block_ids:
        preds.append(Predicate("graspable", [oid]))

    scene = Scene(objects=objects, predicates=preds, table_bounds=TABLE,
                  notes=f"generated seed={seed} kind={kind} diff={difficulty}")
    return EvalTask(scene=scene, goal=goal, goal_predicates=goal_predicates,
                    kind=kind, difficulty=difficulty)


def generate_suite(n: int, base_seed: int = 1000) -> List[EvalTask]:
    return [generate(base_seed + i) for i in range(n)]
