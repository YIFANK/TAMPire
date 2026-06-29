"""Milestone B: EXECUTE a TAMPire plan in the real RoboCasa env and score it with
RoboCasa's NATIVE success check, producing a video.

Manipulation is abstracted (a "magic"/oracle gripper, exactly like the Tier-1
MuJoCo runner): pick/place move the object along a smooth lift-transport-lower
path; open/close actuate a fixture's real door joint. This is NOT full PandaOmron
whole-body control — it validates that the *plan* achieves the task's native goal
in real RoboCasa physics, and renders it. Full arm control is the open follow-on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..schemas import Plan


def _interior_centroid(fixture) -> np.ndarray:
    """Centroid of a fixture's interior region (the 'inside' the success check uses)."""
    sites = fixture.get_int_sites(relative=False)
    p0, px, py, pz = list(sites.values())[0]
    c = (np.asarray(p0) + np.asarray(px) + np.asarray(py) + np.asarray(pz)) / 4.0
    return c + np.array([0.0, 0.0, 0.03])


def target_fixture(env, target_id: str):
    """Map a perceived target id (e.g. 'sink', 'cabinet') onto the env's fixture."""
    t = target_id.lower()
    for key, attr in (("sink", "sink"), ("cab", "cab"), ("microwave", "microwave"),
                      ("oven", "oven"), ("drawer", "drawer"), ("fridge", "fridge")):
        if key in t and hasattr(env, attr):
            return getattr(env, attr)
    for attr in ("sink", "cab", "microwave", "oven", "drawer", "fridge"):
        if hasattr(env, attr):
            return getattr(env, attr)
    return None


@dataclass
class ExecResult:
    native_success: bool
    frames: List[str] = field(default_factory=list)
    gif: Optional[str] = None
    log: List[str] = field(default_factory=list)


class MagicExecutor:
    def __init__(self, env, fixture, *, obj_joint="obj_joint0", obj_name="obj",
                 camera="robot0_agentview_left", w=640, h=480,
                 out_prefix="runs/rc_exec", render=True):
        self.env, self.fx, self.sim = env, fixture, env.sim
        self.obj_joint, self.obj_name = obj_joint, obj_name
        self.camera, self.w, self.h = camera, w, h
        self.out_prefix, self.render_on = out_prefix, render
        self.frames: List[str] = []
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        self._fi = 0

    # ---- low-level ----
    def _obj_pos(self) -> np.ndarray:
        return np.array(self.sim.data.get_joint_qpos(self.obj_joint)[:3])

    def _set_obj(self, xyz) -> None:
        self.sim.data.set_joint_qpos(self.obj_joint, np.concatenate([xyz, [1, 0, 0, 0]]))
        self.sim.data.set_joint_qvel(self.obj_joint, np.zeros(6))
        self.sim.forward()

    def _snap(self) -> None:
        if not self.render_on:
            return
        from PIL import Image
        img = self.sim.render(width=self.w, height=self.h, camera_name=self.camera)[::-1]
        p = f"{self.out_prefix}_{self._fi:03d}.png"
        Image.fromarray(np.asarray(img)).save(p)
        self.frames.append(p)
        self._fi += 1

    def _glide(self, a, b, n=10) -> None:
        for t in np.linspace(0, 1, n):
            self._set_obj(a + (b - a) * t)
            self._snap()

    # ---- primitives (magic) ----
    def hold_still(self, k=4):
        for _ in range(k):
            self._snap()

    def open_fixture(self):
        if hasattr(self.fx, "open_door"):
            self.fx.open_door(self.env)
            self.sim.forward()
        self.hold_still(3)

    def close_fixture(self):
        if hasattr(self.fx, "close_door"):
            self.fx.close_door(self.env)
            self.sim.forward()
        self.hold_still(3)

    def pick_place(self):
        start = self._obj_pos()
        interior = _interior_centroid(self.fx)
        lift_z = max(start[2], interior[2]) + 0.22
        p_lift = np.array([start[0], start[1], lift_z])
        p_over = np.array([interior[0], interior[1], lift_z])
        self._glide(start, p_lift, 8)       # pick: straight up
        self._glide(p_lift, p_over, 12)     # transport over the target
        self._glide(p_over, interior, 8)    # place: lower in
        for _ in range(60):                 # let physics settle it
            self.sim.step()
        self._snap()


def execute_plan(env, fixture, plan: Plan, *, out_prefix="runs/rc_exec",
                 camera="robot0_agentview_left", render=True) -> ExecResult:
    ex = MagicExecutor(env, fixture, camera=camera, out_prefix=out_prefix, render=render)
    log: List[str] = []
    ex._snap()                              # initial frame
    placed = False
    for s in plan.steps:
        if s.action == "open":
            ex.open_fixture(); log.append(f"open({fixture and 'fixture'})")
        elif s.action == "close":
            ex.close_fixture(); log.append("close(fixture)")
        elif s.action == "place" and not placed:
            ex.pick_place(); placed = True; log.append("pick+place -> interior")
        # pick / move_to / gripper ops are folded into the pick_place glide

    native = bool(env._check_success())
    gif = None
    if render and len(ex.frames) > 1:
        from PIL import Image
        imgs = [Image.open(f).convert("RGB") for f in ex.frames]
        gif = f"{out_prefix}.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=120, loop=0)
    return ExecResult(native_success=native, frames=ex.frames, gif=gif, log=log)
