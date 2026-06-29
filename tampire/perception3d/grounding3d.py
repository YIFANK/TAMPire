"""Turn multi-agent vision pose estimates into a TAMPire Scene with geometric
predicates — the bridge from pixels to symbolic+metric state that TAMP plans over.

Predicates are derived geometrically (deterministically) from the estimated 3D, not
asked from an LLM: stacks from xy-clusters + stack_level, containment from bowl
footprints, clear/graspable/container from the resulting relations.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..schemas import Predicate, Scene, WorldObject
from .estimator import PoseEstimate

_BLOCK_SIZE = (0.04, 0.04, 0.04)
_BOWL_SIZE = (0.12, 0.12, 0.05)
_BOWL_RADIUS = 0.07     # xy radius counted as "inside" a bowl
_STACK_XY = 0.05        # xy distance to consider two objects vertically aligned


def _is_bowl(pe: PoseEstimate) -> bool:
    return pe.category in ("bowl", "cup", "plate", "tray")


def build_scene(estimates: Dict[str, PoseEstimate],
                table_bounds=(-0.4, -0.3, 0.4, 0.3)) -> Scene:
    objs: List[WorldObject] = []
    for pe in estimates.values():
        bowl = _is_bowl(pe)
        objs.append(WorldObject(
            id=pe.object_id, category=pe.category, color=pe.color,
            position=(float(pe.xyz[0]), float(pe.xyz[1]), float(pe.xyz[2])),
            size=_BOWL_SIZE if bowl else _BLOCK_SIZE,
            affordances=(["container", "support"] if bowl else ["graspable", "stackable"]),
        ))

    preds: List[Predicate] = []
    blocks = [o for o in objs if "graspable" in o.affordances]
    bowls = [o for o in objs if "container" in o.affordances]

    # supports: for each block, is it stacked on another block (nearest lower one
    # within xy), inside a bowl, or on the table?
    support: Dict[str, str] = {}
    for b in blocks:
        bx, by, bz = b.position
        in_bowl = next((w for w in bowls
                        if np.hypot(bx - w.position[0], by - w.position[1]) < _BOWL_RADIUS), None)
        below = [o for o in blocks if o.id != b.id and o.position[2] < bz - 0.01
                 and np.hypot(bx - o.position[0], by - o.position[1]) < _STACK_XY]
        below.sort(key=lambda o: bz - o.position[2])
        if in_bowl is not None and not below:
            support[b.id] = in_bowl.id
            preds.append(Predicate("in", [b.id, in_bowl.id]))
        elif below:
            support[b.id] = below[0].id
            preds.append(Predicate("on", [b.id, below[0].id]))
        else:
            support[b.id] = "table"
            preds.append(Predicate("on", [b.id, "table"]))

    # clear = nothing is supported by it
    supported_targets = set(support.values())
    for o in objs:
        if o.id not in supported_targets:
            preds.append(Predicate("clear", [o.id]))
    for b in blocks:
        preds.append(Predicate("graspable", [b.id]))
    for w in bowls:
        preds.append(Predicate("container", [w.id]))

    return Scene(objects=objs, predicates=preds, table_bounds=table_bounds,
                 notes="grounded from multi-agent vision pose estimates")
