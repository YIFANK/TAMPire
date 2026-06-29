"""Milestone B (real): drive RoboCasa's PandaOmron with CLOSED-LOOP OSC to actually
grasp and place — no teleport. The 12-D whole-body action is [arm OSC(6), gripper(1),
base(3), torso(1)]; we keep the base/torso still and servo the arm end-effector to
targets with a P-controller, then close the gripper for a real friction grasp.

Scored by RoboCasa's NATIVE _check_success(). This is the genuine arm-control version
of Milestone B (the magic-gripper one is execute.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..schemas import Plan
from .execute import _interior_centroid, target_fixture


@dataclass
class ArmResult:
    native_success: bool
    grasped: bool
    frames: List[str] = field(default_factory=list)
    gif: Optional[str] = None
    log: List[str] = field(default_factory=list)


class RealArmExecutor:
    def __init__(self, env, *, obj_name="obj", camera="robot0_agentview_left",
                 w=640, h=480, out_prefix="runs/rc_arm", render=True):
        self.env, self.obj_name = env, obj_name
        self.camera, self.w, self.h = camera, w, h
        self.out_prefix, self.render_on = out_prefix, render
        self.frames: List[str] = []
        self._fi = 0
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # ---- state ----
    def _obs(self):
        return self.env._get_observations()

    def _eef(self):
        return np.array(self._obs()["robot0_eef_pos"])

    def _obj(self):
        return np.array(self._obs()[f"{self.obj_name}_pos"])

    def _fingers(self):
        return np.array(self._obs()["robot0_gripper_qpos"])

    # ---- rendering ----
    def _snap(self, every_ok=True):
        if not self.render_on:
            return
        from PIL import Image
        img = self.env.sim.render(width=self.w, height=self.h, camera_name=self.camera)[::-1]
        p = f"{self.out_prefix}_{self._fi:03d}.png"
        Image.fromarray(np.asarray(img)).save(p)
        self.frames.append(p)
        self._fi += 1

    # ---- closed-loop control (base/torso held at 0) ----
    def _servo(self, target, grip, *, kp=3.0, steps=200, tol=0.006, snap_every=8):
        ad = self.env.action_dim
        for i in range(steps):
            d = np.asarray(target) - self._eef()
            a = np.zeros(ad)
            a[0:3] = np.clip(kp * d, -1, 1)
            a[6] = grip
            self.env.step(a)
            if snap_every and i % snap_every == 0:
                self._snap()
            if np.linalg.norm(d) < tol:
                break

    def _grip(self, g, n=130, snap_every=30):
        ad = self.env.action_dim
        for i in range(n):
            a = np.zeros(ad)
            a[6] = g
            self.env.step(a)
            if snap_every and i % snap_every == 0:
                self._snap()

    # ---- skills ----
    def pick(self) -> bool:
        o = self._obj()
        self._servo(o + [0, 0, 0.12], -1.0, kp=3)          # approach above, open
        self._servo([o[0], o[1], o[2] - 0.004], -1.0, kp=2, tol=0.004)  # descend onto it
        self._grip(1.0, 130)                                # close (friction grasp)
        self._servo(self._eef() + [0, 0, 0.28], 1.0, kp=3)  # lift
        return bool(self._obj()[2] > o[2] + 0.05)

    def place(self, dest_xyz) -> None:
        dest = np.asarray(dest_xyz, float)
        cur = self._eef()
        carry_z = max(cur[2], dest[2] + 0.15)
        self._servo([cur[0], cur[1], carry_z], 1.0, kp=3)       # raise
        self._servo([dest[0], dest[1], carry_z], 1.0, kp=3)     # over the target
        self._servo([dest[0], dest[1], dest[2] + 0.06], 1.0, kp=2)  # lower in
        self._grip(-1.0, 40)                                    # release
        self._servo([dest[0], dest[1], carry_z + 0.05], -1.0, kp=3)  # retract -> gripper far


def execute_plan_real(env, fixture, plan: Plan, *, out_prefix="runs/rc_arm",
                      camera="robot0_agentview_left", render=True) -> ArmResult:
    ex = RealArmExecutor(env, camera=camera, out_prefix=out_prefix, render=render)
    log: List[str] = []
    ex._snap()
    grasped = False
    for s in plan.steps:
        if s.action == "open" and hasattr(fixture, "open_door"):
            fixture.open_door(env); env.sim.forward(); ex._snap(); log.append("open(fixture)")
        elif s.action == "pick":
            grasped = ex.pick(); log.append(f"pick -> grasped={grasped}")
        elif s.action == "place":
            ex.place(_interior_centroid(fixture)); log.append("place -> interior")
    native = bool(env._check_success())
    gif = None
    if render and len(ex.frames) > 1:
        from PIL import Image
        imgs = [Image.open(f).convert("RGB") for f in ex.frames]
        gif = f"{out_prefix}.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    return ArmResult(native_success=native, grasped=grasped, frames=ex.frames, gif=gif, log=log)
