"""Convex-hull collision checking on the per-object meshes from the depth branch.

TiPToP uses convex-hull completion and finds the (over-approximate) hulls
"surprisingly sufficient" for rearrangement. We do the same: a placement is
collision-free if the placed object's hull, moved to the target, doesn't intersect
any other object's hull. Convex-convex intersection is tested by linear programming
(a separating point exists iff the hulls overlap) — exact for convex sets, no fcl.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _hulls_intersect(A: np.ndarray, B: np.ndarray) -> bool:
    """True if convex hulls of point sets A, B (each (n,3)) intersect.

    Feasibility LP: find weights a>=0 (sum 1), b>=0 (sum 1) with A^T a = B^T b,
    i.e. a common point. Solved as a tiny linear program via scipy.
    """
    from scipy.optimize import linprog

    na, nb = len(A), len(B)
    # variables: [a (na), b (nb)]. Equality: A^T a - B^T b = 0 (3 rows);
    # sum a = 1; sum b = 1. Bounds >= 0. Trivial objective (feasibility).
    Aeq = np.zeros((5, na + nb))
    Aeq[0:3, :na] = A.T
    Aeq[0:3, na:] = -B.T
    Aeq[3, :na] = 1.0
    Aeq[4, na:] = 1.0
    beq = np.array([0, 0, 0, 1, 1.0])
    res = linprog(c=np.zeros(na + nb), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)] * (na + nb), method="highs")
    return bool(res.success)


def hull_at(hull_vertices: np.ndarray, new_center: np.ndarray, old_center: np.ndarray) -> np.ndarray:
    """Translate a hull so its centroid moves from old_center to new_center."""
    return hull_vertices + (np.asarray(new_center) - np.asarray(old_center))


def placement_collisions(placed_hull: np.ndarray, other_hulls: Dict[str, np.ndarray],
                         *, ignore: Optional[List[str]] = None) -> List[str]:
    """Return ids whose hull intersects the placed object's hull."""
    ignore = set(ignore or [])
    hits = []
    for oid, hull in other_hulls.items():
        if oid in ignore or hull is None:
            continue
        if _hulls_intersect(placed_hull, hull):
            hits.append(oid)
    return hits


def check_place(perceived, obj_id: str, target_id: str) -> List[str]:
    """TAMP placement feasibility on PERCEIVED meshes: would putting `obj_id` on/in
    `target_id` collide with any other object? Returns the colliding ids ([] = free).

    `perceived` is a perception3d.perceive.PerceivedScene. We translate obj's convex
    hull to where it would land (on top of a block, or into a container) and test it
    against every other object's hull.
    """
    hulls = perceived.hulls()
    scene = perceived.scene
    obj, target = scene.by_id(obj_id), scene.by_id(target_id)
    if obj is None or obj_id not in hulls:
        return []
    oc = np.array(obj.position)
    if target_id == "table":
        dest = np.array([oc[0], oc[1], (obj.size[2] / 2)])
    else:
        tc = np.array(target.position)
        if "container" in target.affordances:          # into a bowl
            dest = np.array([tc[0], tc[1], tc[2]])
        else:                                          # onto a block
            dest = np.array([tc[0], tc[1], tc[2] + target.size[2] / 2 + obj.size[2] / 2])
    placed = hull_at(hulls[obj_id], dest, oc)
    return placement_collisions(placed, hulls, ignore=[obj_id, target_id])
