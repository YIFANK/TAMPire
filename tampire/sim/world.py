"""Symbolic + 2.5D geometric world model.

Tracks where every object is (on what / in what / held) and supports applying a
single PlanStep, returning (ok, reason). No physics — just preconditions and
simple geometry. That is deliberate: it's fast, deterministic, and the failure
*reason* is exactly what the debate council needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..schemas import Predicate, PlanStep, Scene, WorldObject

# Robot base + reach (metres). Single arm at the near edge of the table.
BASE_XY: Tuple[float, float] = (0.0, -0.45)
REACH_RADIUS: float = 0.95


@dataclass
class World:
    objects: Dict[str, WorldObject]
    table_bounds: Tuple[float, float, float, float]
    # relations
    on: Dict[str, str] = field(default_factory=dict)        # obj -> support (obj rests on support)
    inside: Dict[str, str] = field(default_factory=dict)    # obj -> container
    holding: Optional[str] = None
    open_state: Dict[str, bool] = field(default_factory=dict)  # openable fixture -> is_open
    base_xy: Tuple[float, float] = BASE_XY                   # mobile base position (mobile manip)
    reach_radius: float = REACH_RADIUS                      # effective arm reach from the base

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_scene(cls, scene: Scene) -> "World":
        objs = {o.id: o for o in scene.objects}
        w = cls(objects=objs, table_bounds=scene.table_bounds)
        # seed relations from predicates; default everything onto the table
        for o in scene.objects:
            w.on.setdefault(o.id, "table")
            if "openable" in o.affordances:
                w.open_state[o.id] = False   # openable fixtures start closed
        for p in scene.predicates:
            if p.negated:
                continue
            if p.name == "on" and len(p.args) == 2:
                w.on[p.args[0]] = p.args[1]
            elif p.name == "in" and len(p.args) == 2:
                w.inside[p.args[0]] = p.args[1]
                w.on.pop(p.args[0], None)
            elif p.name in ("open", "closed") and p.args:
                w.open_state[p.args[0]] = (p.name == "open")
            elif p.name == "base_at" and len(p.args) >= 2:
                try:
                    w.base_xy = (float(p.args[0]), float(p.args[1]))
                except (TypeError, ValueError):
                    pass
            elif p.name == "reach_radius" and p.args:
                try:
                    w.reach_radius = float(p.args[0])
                except (TypeError, ValueError):
                    pass
        return w

    # ---- queries ----------------------------------------------------------
    def exists(self, oid: str) -> bool:
        return oid == "table" or oid in self.objects

    def is_clear(self, oid: str) -> bool:
        """Nothing stacked on oid and nothing inside (for blocks)."""
        if oid == "table":
            return True
        if any(s == oid for s in self.on.values()):
            return False
        if any(c == oid for c in self.inside.values()):
            return False
        return True

    def has_affordance(self, oid: str, aff: str) -> bool:
        o = self.objects.get(oid)
        return bool(o and aff in o.affordances)

    def reachable(self, oid: str) -> bool:
        if oid == "table":
            return True
        o = self.objects.get(oid)
        if not o:
            return False
        x, y = o.position[0], o.position[1]
        # reach is measured from the CURRENT mobile-base position (move_base changes it)
        dist = math.hypot(x - self.base_xy[0], y - self.base_xy[1])
        if dist > self.reach_radius:
            return False
        xmin, ymin, xmax, ymax = self.table_bounds
        margin = 0.05
        return (xmin - margin) <= x <= (xmax + margin) and (ymin - margin) <= y <= (ymax + margin)

    # ---- transition -------------------------------------------------------
    def apply(self, step: PlanStep) -> Tuple[bool, str]:
        a = step.action
        args = step.args
        if a == "move_to":
            if not args or not self.exists(args[0]):
                return False, f"move_to references unknown object '{args[0] if args else ''}'"
            if not self.reachable(args[0]):
                return False, f"{args[0]} is out of the arm's reach"
            return True, ""

        if a in ("open_gripper", "close_gripper"):
            return True, ""

        if a == "move_base":
            # drive the mobile base next to `target` so it falls within arm reach
            if not args:
                return False, "move_base missing target argument"
            tgt = args[0]
            if not self.exists(tgt):
                return False, f"move_base to unknown target '{tgt}'"
            if tgt == "table":
                return True, ""
            o = self.objects.get(tgt)
            if o is not None:
                self.base_xy = (o.position[0], o.position[1])
            return True, ""

        if a in ("open", "close"):
            if not args:
                return False, f"{a} missing fixture argument"
            fx = args[0]
            if not self.exists(fx):
                return False, f"cannot {a} unknown fixture '{fx}'"
            if not self.has_affordance(fx, "openable"):
                return False, f"{fx} is not openable"
            if not self.reachable(fx):
                return False, f"{fx} is out of the arm's reach"
            self.open_state[fx] = (a == "open")
            return True, ""

        if a == "pick":
            if not args:
                return False, "pick missing object argument"
            obj = args[0]
            if self.holding is not None:
                return False, f"cannot pick {obj}: gripper already holding {self.holding}"
            if not self.exists(obj) or obj == "table":
                return False, f"cannot pick unknown object '{obj}'"
            if not self.has_affordance(obj, "graspable"):
                return False, f"{obj} is not graspable"
            if not self.is_clear(obj):
                blocker = next((o for o, s in self.on.items() if s == obj), "something")
                return False, f"cannot pick {obj}: {blocker} is on top of it (not clear)"
            if not self.reachable(obj):
                return False, (f"{obj} is out of the arm's reach from the current base "
                               f"position; move_base({obj}) first")
            # effect
            self.on.pop(obj, None)
            self.inside.pop(obj, None)
            self.holding = obj
            return True, ""

        if a == "place":
            if len(args) < 2:
                return False, "place requires (object, target)"
            obj, target = args[0], args[1]
            if self.holding != obj:
                held = self.holding or "nothing"
                return False, f"cannot place {obj}: gripper is holding {held}"
            if not self.exists(target) or target == obj:
                return False, f"cannot place onto invalid target '{target}'"
            if not self.reachable(target):
                return False, (f"target {target} is out of the arm's reach from the current "
                               f"base position; move_base({target}) first")
            is_container = self.has_affordance(target, "container")
            if is_container:
                if self.has_affordance(target, "openable") and not self.open_state.get(target, False):
                    return False, f"cannot place {obj} in {target}: {target} is closed (open it first)"
                self.inside[obj] = target
                self.on.pop(obj, None)
            else:
                if target != "table" and not self.is_clear(target):
                    blocker = next((o for o, s in self.on.items() if s == target), "something")
                    return False, f"cannot stack {obj} on {target}: {blocker} already on it (not clear)"
                self.on[obj] = target
                self.inside.pop(obj, None)
            self.holding = None
            return True, ""

        return False, f"unknown action '{a}'"

    # ---- goal check -------------------------------------------------------
    def satisfies(self, goal_predicates: List[Predicate]) -> bool:
        for p in goal_predicates:
            if p.name == "on":
                ok = self.on.get(p.args[0]) == p.args[1]
            elif p.name == "in":
                ok = self.inside.get(p.args[0]) == p.args[1]
            elif p.name in ("open", "closed"):
                is_open = self.open_state.get(p.args[0], False)
                ok = is_open if p.name == "open" else (not is_open)
            else:
                ok = True  # unknown goal predicate -> don't block
            if p.negated:
                ok = not ok
            if not ok:
                return False
        return True

    def state_predicates(self) -> List[Predicate]:
        preds: List[Predicate] = []
        for o, s in self.on.items():
            preds.append(Predicate("on", [o, s]))
        for o, c in self.inside.items():
            preds.append(Predicate("in", [o, c]))
        if self.holding:
            preds.append(Predicate("holding", [self.holding]))
        return preds
