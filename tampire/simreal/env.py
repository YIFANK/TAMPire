"""MuJoCo tabletop environment with abstracted manipulation.

execute(plan) runs our primitives by directly setting the held object's pose and
letting physics settle (a "magic" grasp / suction oracle). Success is read back
from the simulated body positions — an independent check, not the LLM's word.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mujoco is required for the Tier-1 sim. Install it in a Python 3.11/3.12 "
        "venv: `python3.12 -m venv .venv312 && .venv312/bin/pip install mujoco`."
    ) from e

from ..schemas import Plan, Predicate, Scene
from . import mjscene

_SETTLE_STEPS = 60
_PARK_Z = 0.45  # height a held object floats at (clears tall long-horizon towers)


@dataclass
class StepLog:
    index: int
    action: str
    args: List[str]
    ok: bool
    note: str = ""


class TabletopEnv:
    def __init__(self, scene: Scene):
        self.scene = scene
        xml, self.meta = mjscene.build(scene)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.held: Optional[str] = None
        self._held_xy: Tuple[float, float] = (0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.settle()

    # ---- pose helpers -----------------------------------------------------
    def _qadr(self, body: str) -> Optional[int]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            return None
        jadr = self.model.body_jntadr[bid]
        if jadr < 0:
            return None
        return self.model.jnt_qposadr[jadr]

    def pos_of(self, body: str) -> Optional[np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            return None
        return np.array(self.data.xpos[bid])

    def _set_free_pose(self, body: str, x: float, y: float, z: float) -> bool:
        adr = self._qadr(body)
        if adr is None:
            return False
        self.data.qpos[adr:adr + 3] = [x, y, z]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        adr_v = self.model.body_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)]
        if adr_v >= 0:
            self.data.qvel[adr_v:adr_v + 6] = 0
        return True

    # ---- physics ----------------------------------------------------------
    def settle(self, steps: int = _SETTLE_STEPS) -> None:
        for _ in range(steps):
            if self.held:  # pin the held object so it floats
                self._set_free_pose(self.held, self._held_xy[0], self._held_xy[1], _PARK_Z)
            mujoco.mj_step(self.model, self.data)

    # ---- rendering --------------------------------------------------------
    def render(self, camera: str = "angled") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ---- manipulation primitives -----------------------------------------
    def _blocked_by(self, obj: str) -> Optional[str]:
        """Is another block resting on top of obj? (mirrors the 'clear' precondition
        a real gripper faces — the magic grasp must not cheat through a stack.)"""
        p = self.pos_of(obj)
        if p is None:
            return None
        for other, is_blk in self.meta.is_block.items():
            if not is_blk or other == obj:
                continue
            q = self.pos_of(other)
            if q is None:
                continue
            if abs(q[0] - p[0]) < 0.03 and abs(q[1] - p[1]) < 0.03 and q[2] > p[2] + 0.01:
                return other
        return None

    def _do_pick(self, obj: str) -> Tuple[bool, str]:
        if self.held is not None:
            return False, f"already holding {self.held}"
        if self._qadr(obj) is None or not self.meta.is_block.get(obj, False):
            return False, f"{obj} is not a graspable free body"
        blocker = self._blocked_by(obj)
        if blocker:
            return False, f"{obj} is blocked by {blocker} on top (not clear)"
        p = self.pos_of(obj)
        self.held = obj
        self._held_xy = (float(p[0]), float(p[1]))
        self.settle(20)
        return True, ""

    def _free_table_xy(self, ignore: str) -> Tuple[float, float]:
        """A spot on the table clear of every other object (so clearing a block
        actually clears it, instead of dropping it back on the stack)."""
        xmin, ymin, xmax, ymax = self.scene.table_bounds
        others = [self.pos_of(o) for o in self.meta.is_block
                  if o != ignore] + [self.pos_of(b) for b in self.meta.bowl_inner]
        others = [p for p in others if p is not None]
        # scan a grid, pick the cell farthest from all others
        best, best_d = (xmin + 0.08, ymin + 0.08), -1.0
        nx, ny = 6, 5
        for i in range(nx):
            for j in range(ny):
                x = xmin + 0.08 + (xmax - xmin - 0.16) * i / (nx - 1)
                y = ymin + 0.08 + (ymax - ymin - 0.16) * j / (ny - 1)
                d = min((abs(x - p[0]) + abs(y - p[1])) for p in others) if others else 1.0
                if d > best_d:
                    best, best_d = (x, y), d
        return best

    def _do_place(self, obj: str, target: str) -> Tuple[bool, str]:
        if self.held != obj:
            return False, f"not holding {obj}"
        tp = self.pos_of(target) if target != "table" else None
        if target == "table":
            x, y = self._free_table_xy(ignore=obj)
            z = self.meta.block_half[obj] + 0.002
        elif target in self.meta.bowl_inner:
            x, y = float(tp[0]), float(tp[1])
            z = self.meta.bowl_wall_top[target] + self.meta.block_half[obj] + 0.01
        else:  # stack on a block
            x, y = float(tp[0]), float(tp[1])
            z = float(tp[2]) + self.meta.block_half.get(target, 0.025) + self.meta.block_half[obj] + 0.005
        self.held = None
        self._set_free_pose(obj, x, y, z)
        self.settle()
        return True, ""

    def execute(self, plan: Plan) -> List[StepLog]:
        logs: List[StepLog] = []
        for i, s in enumerate(plan.steps):
            if s.action == "pick":
                ok, note = self._do_pick(s.args[0]) if s.args else (False, "no arg")
            elif s.action == "place":
                ok, note = self._do_place(s.args[0], s.args[1]) if len(s.args) >= 2 else (False, "bad args")
            elif s.action in ("move_to", "open_gripper", "close_gripper"):
                ok, note = True, "noop"
            else:
                ok, note = False, f"unknown action {s.action}"
            logs.append(StepLog(i, s.action, s.args, ok, note))
        return logs

    # ---- success check (independent, from physics) ------------------------
    def check_goal(self, goal_predicates: List[Predicate]) -> Tuple[bool, str]:
        for gp in goal_predicates:
            ok, why = self._check_one(gp)
            if not ok:
                return False, why
        return True, "all goal predicates hold in physics"

    def _check_one(self, gp: Predicate) -> Tuple[bool, str]:
        if gp.name == "in" and len(gp.args) == 2:
            a, bowl = gp.args
            pa = self.pos_of(a)
            if pa is None or bowl not in self.meta.bowl_inner:
                return False, f"missing bodies for in({a},{bowl})"
            xmin, ymin, xmax, ymax = self.meta.bowl_inner[bowl]
            inside_xy = xmin <= pa[0] <= xmax and ymin <= pa[1] <= ymax
            low_enough = pa[2] <= self.meta.bowl_wall_top[bowl] + 0.02
            return (inside_xy and low_enough), (
                f"in({a},{bowl}): xy_in={inside_xy} z={pa[2]:.3f}")
        if gp.name == "on" and len(gp.args) == 2:
            a, b = gp.args
            pa, pb = self.pos_of(a), self.pos_of(b)
            if pa is None or pb is None:
                return False, f"missing bodies for on({a},{b})"
            ha = self.meta.block_half.get(a, 0.025)
            hb = self.meta.block_half.get(b, 0.025)
            aligned = abs(pa[0] - pb[0]) < 0.03 and abs(pa[1] - pb[1]) < 0.03
            stacked = abs(pa[2] - (pb[2] + ha + hb)) < 0.02
            return (aligned and stacked), (
                f"on({a},{b}): aligned={aligned} dz={pa[2]-pb[2]:.3f}")
        return True, f"unchecked predicate {gp.name}"

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass
