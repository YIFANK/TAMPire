"""A custom robosuite env: a REAL Panda arm + N colored cubes, for executing the
long-horizon buried-base tower with actual scripted grasps (not magic teleport).

Subclasses robosuite's Stack so we inherit the Panda/arena/controller plumbing,
but builds N BoxObjects and starts them deterministically as one stacked column
(the buried-base start state). The scripted OSC pick_place is generalized to N
cubes with explicit table-parking spots so disassembly doesn't pile cubes up.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

OPEN, CLOSE = -1.0, 1.0

CUBE_RGBA = {
    "red": [0.85, 0.18, 0.18, 1], "green": [0.17, 0.62, 0.29, 1],
    "blue": [0.19, 0.40, 0.84, 1], "yellow": [0.95, 0.76, 0.05, 1],
    "orange": [0.91, 0.45, 0.10, 1], "purple": [0.56, 0.27, 0.68, 1],
    "cyan": [0.10, 0.71, 0.77, 1],
}


class TowerEnv(Stack):
    def __init__(self, *, start_order: List[str], column_xy=(0.0, 0.0),
                 cube_half: float = 0.025, **kwargs):
        self._start_order = list(start_order)      # colors bottom->top at start
        self._column_xy = column_xy
        self._cube_half = cube_half
        self._cubes: List[BoxObject] = []
        self._cube_by_color: Dict[str, BoxObject] = {}
        self._cube_body: Dict[str, int] = {}
        self.obs: Optional[dict] = None
        self.frames: List[np.ndarray] = []
        super().__init__(**kwargs)

    # ---- model: N cubes instead of 2 ----
    def _load_model(self):
        # mirror Stack._load_model, but with our N colored cubes + no random sampler
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        ManipulationEnv._load_model(self)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        h = self._cube_half
        self._cubes, self._cube_by_color = [], {}
        for color in self._start_order:
            c = BoxObject(name=f"cube_{color}", size_min=[h, h, h], size_max=[h, h, h],
                          rgba=CUBE_RGBA.get(color, [0.6, 0.6, 0.6, 1]))
            self._cubes.append(c)
            self._cube_by_color[color] = c
        # satisfy inherited Stack code that references cubeA/cubeB
        self.cubeA, self.cubeB = self._cubes[0], self._cubes[min(1, len(self._cubes) - 1)]
        self.placement_initializer = None
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self._cubes,
        )

    def _setup_references(self):
        super()._setup_references()
        self._cube_body = {col: self.sim.model.body_name2id(c.root_body)
                           for col, c in self._cube_by_color.items()}

    def _reset_internal(self):
        # bypass Stack's random sampler; place cubes as one deterministic column
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        ManipulationEnv._reset_internal(self)
        cx, cy = self._column_xy
        h = self._cube_half
        top = self.table_offset[2]
        for level, color in enumerate(self._start_order):
            z = top + h + level * 2 * h
            cube = self._cube_by_color[color]
            self.sim.data.set_joint_qpos(
                cube.joints[0], np.array([cx, cy, z, 1, 0, 0, 0]))

    def _check_success(self):  # we score the tower geometrically in the runner
        return False

    def reward(self, action=None):
        return 0.0

    # ---- state ----
    def boot(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            np.random.seed(seed)
        self.obs = self.reset()
        self.frames = []
        return self.obs

    def cube_pos(self, color: str) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self._cube_body[color]])

    def eef(self) -> np.ndarray:
        return np.array(self.obs["robot0_eef_pos"])

    @property
    def table_top(self) -> float:
        return float(self.table_offset[2])

    # ---- rendering ----
    def render_named(self, name: str, img: int = 320) -> np.ndarray:
        rgb = self.sim.render(width=img, height=img, camera_name=name)
        return np.flipud(rgb).copy()

    def capture(self, camera: str = "frontview", img: int = 320) -> None:
        self.frames.append(self.render_named(camera, img))

    def save_gif(self, path: str, fps: float = 14) -> Optional[str]:
        if not self.frames:
            return None
        from PIL import Image
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        return path

    # ---- low-level OSC control (mirrors tier2 StackEnv) ----
    def _move(self, target, grip: float, steps: int = 120, tol: float = 0.006,
              kp: float = 8.0, cam: str = "frontview", capture_every: int = 6) -> None:
        for i in range(steps):
            d = np.asarray(target) - self.eef()
            a = np.zeros(7)
            a[:3] = np.clip(kp * d, -1, 1)
            a[6] = grip
            self.obs, _, _, _ = self.step(a)
            if capture_every and i % capture_every == 0:
                self.capture(cam)
            if np.linalg.norm(d) < tol:
                break

    def _grip(self, g: float, n: int = 25, cam: str = "frontview") -> None:
        for _ in range(n):
            self.obs, _, _, _ = self.step(np.array([0, 0, 0, 0, 0, 0, float(g)]))
        self.capture(cam)

    def pick_place(self, src_color: str, dst, *, park_xy=None,
                   cam: str = "frontview", capture_every: int = 6) -> None:
        """Pick cube `src_color`; place on `dst` (a color) or on the table.

        dst == "table": placed at park_xy (an explicit free spot).
        dst == <color>: stacked on that cube.
        """
        h = self._cube_half
        A = self.cube_pos(src_color).copy()
        self._move(A + [0, 0, 0.12], OPEN, cam=cam, capture_every=capture_every)
        self._move(A + [0, 0, 0.0], OPEN, kp=5, cam=cam, capture_every=capture_every)
        self._grip(CLOSE, 25, cam=cam)
        self._move(A + [0, 0, 0.22], CLOSE, cam=cam, capture_every=capture_every)
        if dst == "table":
            x, y = park_xy
            place_z = self.table_top + h + 0.004
        else:
            B = self.cube_pos(dst).copy()
            x, y = float(B[0]), float(B[1])
            place_z = float(B[2]) + 2 * h + 0.004
        self._move([x, y, place_z + 0.18], CLOSE, cam=cam, capture_every=capture_every)
        self._move([x, y, place_z + 0.02], CLOSE, kp=5, cam=cam, capture_every=capture_every)
        self._grip(OPEN, 12, cam=cam)
        self._move([x, y, place_z + 0.20], OPEN, cam=cam, capture_every=capture_every)
