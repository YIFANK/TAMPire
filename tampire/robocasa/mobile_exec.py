"""Vision-based mobile manipulation in RoboCasa — the full closed loop.

  see_multicam : Gemma localizes the object in SEVERAL camera views; each pixel is
                 back-projected onto the counter plane (every camera's pose is known,
                 so all views land in one world frame) and fused by MEDIAN — robust to
                 a view that locks onto a distractor.
  drive_smooth : animated base repositioning — the base joints are interpolated to the
                 target (with reset_goal each step so the controllers don't fight it),
                 so the base visibly glides across the kitchen. (Kinematic, not torque-
                 driven: RoboCasa's mobile-base velocity controller is a black box.)
  grasp/place  : REAL OSC torque control of the arm + gripper.

The drive target and grasp target come from VISION, not privileged obj_pos. Scored by
RoboCasa's NATIVE _check_success.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

COUNTER_Z = 0.95
CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_agentview_center")
_SYS = ("Kitchen image, normalized coords 0-1000 (x=left->right, y=top->bottom). Find the small "
        "graspable food item on the counter the robot must pick up (ignore anything already in the "
        'sink/basin). Reply ONLY JSON {"x":<0-1000>,"y":<0-1000>} at the object center, or '
        '{"x":-1,"y":-1} if no such object is clearly visible.')


def see_multicam(env, instr):
    """Fuse per-camera Gemma localizations into one world-xy on the counter plane."""
    from PIL import Image
    import mujoco
    from .. import llm
    from ..perception3d import camera as C
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    mujoco.mj_forward(m, d)
    cands = []
    for cam in CAMS:
        try:
            img = np.flipud(sim.render(width=640, height=480, camera_name=cam)).copy()
            p = os.path.join(tempfile.mkdtemp(), "v.png")
            Image.fromarray(img).save(p)
            a, _ = llm.chat_json([llm.sys(_SYS), llm.user_with_image(f"Task: {instr}", p)],
                                 label=f"see:{cam}", temperature=0)
            if float(a.get("x", -1)) < 0:
                continue
            cam_m = C.from_mujoco(m, d, cam, 640, 480)
            pt = cam_m.backproject_to_plane(a["x"] / 1000 * 640, a["y"] / 1000 * 480, COUNTER_Z)
            cands.append(pt)
        except Exception:
            continue
    if not cands:
        return None
    cands = np.array(cands)
    med = np.median(cands, axis=0)
    # reject views >15cm from the median, average the consensus
    keep = cands[np.linalg.norm(cands[:, :2] - med[:2], axis=1) < 0.15]
    return keep.mean(axis=0) if len(keep) else med


class MobileArm:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.cc = env.robots[0].composite_controller
        self.FWD = "mobilebase0_joint_mobile_forward"
        self.SIDE = "mobilebase0_joint_mobile_side"
        self.frames = []
        b = self._base()
        f, s = self._qp(self.FWD), self._qp(self.SIDE)
        self._setj(self.FWD, f + 1); dF = self._base() - b; self._setj(self.FWD, f)
        self._setj(self.SIDE, s + 1); dS = self._base() - b; self._setj(self.SIDE, s)
        self.Binv = np.linalg.inv(np.column_stack([dF, dS]))

    # state
    def _base(self): return np.array(self.sim.data.get_body_xpos("mobilebase0_base"))[:2]
    def _eef(self): return np.array(self.env._get_observations()["robot0_eef_pos"])
    def _qp(self, j): return float(self.sim.data.get_joint_qpos(j))
    def _setj(self, j, v): self.sim.data.set_joint_qpos(j, v); self.sim.forward()
    def _rg(self):
        for p in self.cc.part_controllers.values():
            if hasattr(p, "reset_goal"):
                p.reset_goal()

    def snap(self, cam="robot0_agentview_left"):
        self.frames.append(np.flipud(self.sim.render(width=560, height=420, camera_name=cam)).copy())

    # base repositioning: iterative full-target teleport + controller resync + settle
    # (the method proven to preserve the subsequent grasp). Each iteration renders, so
    # the base visibly steps to the target across a handful of frames.
    def drive_smooth(self, tx, ty, iters=8):
        for _ in range(iters):
            dd = self.Binv @ (np.array([tx, ty]) - self._base())
            self._setj(self.FWD, self._qp(self.FWD) + dd[0])
            self._setj(self.SIDE, self._qp(self.SIDE) + dd[1])
            self.sim.data.qvel[:] = 0; self.sim.forward(); self._rg()
            for _ in range(8):
                self.env.step(np.zeros(self.env.action_dim))
            self.snap()
            if np.linalg.norm(np.array([tx, ty]) - self._base()) < 0.01:
                break

    # ---- REAL velocity driving (no teleport): command the mobile-base velocity
    # controller (action[8]=forward, action[9]=yaw) so the base physically rolls. -----
    def _vel(self, a8=0.0, a9=0.0):
        a = np.zeros(self.env.action_dim)
        a[8] = a8; a[9] = a9
        self.env.step(a)

    def calib_forward(self, pulse=0.4, n=12):
        """Measure the base's body-forward direction in world coords (a small real roll)."""
        p0 = self._base()
        for i in range(n):
            self._vel(a8=pulse)
            if i % 3 == 0:
                self.snap()
        f = self._base() - p0
        self.fwd = f / (np.linalg.norm(f) + 1e-9)
        return self.fwd

    def drive_back(self, dist=0.55, speed=0.7, maxk=120):
        """Reverse along body-forward to open up a standoff gap (real physics)."""
        p0 = self._base()
        for k in range(maxk):
            if np.linalg.norm(self._base() - p0) >= dist:
                break
            self._vel(a8=-speed)
            if k % 3 == 0:
                self.snap()

    def drive_to(self, obj_xy, reach=0.34, maxk=200):
        """Drive forward (closed-loop on forward distance to the object) until the arm
        is within reach. Lateral residual is handled by the arm's OSC reach."""
        obj_xy = np.asarray(obj_xy, dtype=float)[:2]
        for k in range(maxk):
            if np.linalg.norm(obj_xy - self._eef()[:2]) < reach:
                break
            fdist = float(np.dot(obj_xy - self._base(), self.fwd))
            self._vel(a8=float(np.clip(2.0 * fdist, -1, 1)))
            if k % 3 == 0:
                self.snap()

    # ---- full unicycle navigation (turn-in-place, then drive) -----------------
    def _yaw(self):
        import math
        w, x, y, z = self.sim.data.get_body_xquat("mobilebase0_base")
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _calib_offset(self):
        """Heading offset between the chassis yaw and the world body-forward axis."""
        import math
        p0 = self._base()
        for _ in range(10):
            self._vel(a8=0.4)
        self._off = math.atan2(*(self._base() - p0)[::-1]) - self._yaw()

    def _heading(self):
        if not hasattr(self, "_off"):
            self._calib_offset()
        return self._yaw() + self._off

    def approach(self, obj_xy, reach=0.30, outer=6):
        """Turn to face the object, drive forward, re-check ARM reachability; repeat
        until the end-effector can reach it. Real velocity control — no teleport."""
        import math
        obj_xy = np.asarray(obj_xy, dtype=float)[:2]
        if not hasattr(self, "_off"):
            self._calib_offset()
        for _ in range(outer):
            if np.linalg.norm(obj_xy - self._eef()[:2]) < reach:
                return True
            # aim a standoff point short of the object so the arm (not the base) finishes
            d = obj_xy - self._base(); dist = np.linalg.norm(d)
            aim = obj_xy - d / (dist + 1e-9) * 0.30
            for _ in range(80):                       # turn in place to face the aim point
                e = aim - self._base()
                he = (math.atan2(e[1], e[0]) - self._heading() + math.pi) % (2 * math.pi) - math.pi
                if abs(he) < math.radians(5):
                    break
                self._vel(a9=float(np.clip(2.5 * he, -1, 1)))
                self.snap()
            for _ in range(120):                      # drive forward toward the aim point
                e = aim - self._base(); h = self._heading()
                fd = e[0] * math.cos(h) + e[1] * math.sin(h)
                if np.linalg.norm(obj_xy - self._eef()[:2]) < reach or fd < 0.05:
                    break
                self._vel(a8=float(np.clip(1.5 * fd, 0.05, 1)))
                self.snap()
        return np.linalg.norm(obj_xy - self._eef()[:2]) < reach

    def move(self, t, steps, kp, grip):
        for i in range(steps):
            e = np.asarray(t) - self._eef()
            a = np.zeros(self.env.action_dim)
            a[0:3] = np.clip(kp * e, -1, 1)
            a[6] = grip
            self.env.step(a)
            if i % 6 == 0:
                self.snap()

    def _grip(self, g, n):
        a = np.zeros(self.env.action_dim)
        a[6] = g
        for _ in range(n):
            self.env.step(a)

    def grasp(self, xyz):
        self.move(xyz + [0, 0, 0.10], 150, 3, -1)
        self.move(xyz + [0, 0, -0.004], 130, 2, -1)
        self._grip(1.0, 60)
        self.move(self._eef() + [0, 0, 0.28], 90, 3, 1)

    def place(self, xyz):
        self.move([xyz[0], xyz[1], self._eef()[2]], 100, 3, 1)
        self.move([xyz[0], xyz[1], xyz[2] + 0.06], 100, 2, 1)
        self._grip(-1.0, 40)
        self.move(self._eef() + [0, 0, 0.2], 60, 3, -1)

    def save_gif(self, path, fps=12):
        from PIL import Image
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return path
