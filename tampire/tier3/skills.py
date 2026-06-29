"""Low-level skill controller: turns TAMPire's high-level plan into the per-step
action chunks RoboLab expects.

A plan of pick/place primitives is expanded into EEF waypoints (approach, grasp,
lift, place, release). Each step we emit a proportional EEF-delta action toward the
current waypoint and advance when reached. This is the same control scheme validated
in the Tier-2 robosuite bridge, here emitting action vectors instead of stepping a sim.

The action layout is configurable (`ActionSpec`) because it must match the registered
RoboLab task's controller. The default is a 7-D differential-IK / OSC pose action
``[dx, dy, dz, drx, dry, drz, gripper]`` — the common Isaac-Lab manipulation convention.
Joint-position tasks (like the Pi0 DROID client) need an IK layer instead; see README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schemas import Plan


@dataclass
class ActionSpec:
    dim: int = 7                # full action vector length
    pos_slice: Tuple[int, int] = (0, 3)
    gripper_index: int = 6
    open_val: float = -1.0
    close_val: float = 1.0
    kp: float = 8.0             # proportional gain on position error
    max_delta: float = 1.0      # clip per-axis


@dataclass
class Waypoint:
    xyz: np.ndarray
    gripper: float
    tol: float = 0.01
    dwell: int = 0             # steps to hold here (for gripper actuation)


@dataclass
class SkillRunner:
    """Per-env state machine over EEF waypoints derived from a plan."""
    spec: ActionSpec
    waypoints: List[Waypoint] = field(default_factory=list)
    idx: int = 0
    dwell_left: int = 0

    # waypoint geometry (metres)
    APPROACH_Z: float = 0.10
    GRASP_DZ: float = 0.0
    LIFT_Z: float = 0.18
    STACK_DZ: float = 0.05     # place height above a target object's centre

    @classmethod
    def from_plan(cls, plan: Plan, poses: Dict[str, np.ndarray], spec: ActionSpec,
                  table_z: float = 0.0) -> "SkillRunner":
        wps: List[Waypoint] = []
        o, c = spec.open_val, spec.close_val
        for s in plan.steps:
            if s.action == "pick" and s.args:
                p = poses.get(s.args[0])
                if p is None:
                    continue
                wps += [
                    Waypoint(p + [0, 0, cls.APPROACH_Z], o),
                    Waypoint(p + [0, 0, cls.GRASP_DZ], o, tol=0.006),
                    Waypoint(p + [0, 0, cls.GRASP_DZ], c, dwell=8),
                    Waypoint(p + [0, 0, cls.LIFT_Z], c),
                ]
            elif s.action == "place" and len(s.args) >= 2:
                tgt = s.args[1]
                if tgt == "table":
                    base = np.array([0.0, -0.15, table_z])
                    place = base + [0, 0, 0.02]
                else:
                    tp = poses.get(tgt)
                    if tp is None:
                        continue
                    base = tp.copy()
                    place = tp + [0, 0, cls.STACK_DZ]
                wps += [
                    Waypoint(base + [0, 0, cls.APPROACH_Z], c),
                    Waypoint(place, c, tol=0.006),
                    Waypoint(place, o, dwell=8),
                    Waypoint(place + [0, 0, cls.LIFT_Z], o),
                ]
        return cls(spec=spec, waypoints=wps)

    @property
    def done(self) -> bool:
        return self.idx >= len(self.waypoints)

    def step(self, eef_pos: np.ndarray) -> np.ndarray:
        """Return one action vector toward the current waypoint."""
        spec = self.spec
        a = np.zeros(spec.dim, dtype=np.float32)
        if self.done:
            a[spec.gripper_index] = self.waypoints[-1].gripper if self.waypoints else spec.open_val
            return a

        wp = self.waypoints[self.idx]
        d = wp.xyz - np.asarray(eef_pos)
        lo, hi = spec.pos_slice
        a[lo:hi] = np.clip(spec.kp * d, -spec.max_delta, spec.max_delta)
        a[spec.gripper_index] = wp.gripper

        reached = float(np.linalg.norm(d)) < wp.tol
        if reached:
            if self.dwell_left == 0 and wp.dwell:
                self.dwell_left = wp.dwell
            if self.dwell_left > 0:
                self.dwell_left -= 1
                a[lo:hi] = 0.0  # hold position while gripper actuates
                if self.dwell_left == 0:
                    self.idx += 1
            else:
                self.idx += 1
        return a
