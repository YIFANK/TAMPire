"""robosuite Stack wrapper with a scripted OSC pick-place skill and frame capture.

The Stack task: cubeA (red) and cubeB (green) on a table; success = cubeA stacked
on cubeB with the gripper released. We expose agentview pixels for perception and
a `pick_place` skill that the planned primitives drive.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "robosuite is required for Tier-2. Install in the 3.12 venv:\n"
        "  .venv312/bin/pip install -r requirements-tier2.txt\n"
        "(needs mujoco==3.3.0; newer mujoco breaks robosuite 1.5)."
    ) from e

OPEN, CLOSE = -1.0, 1.0

# robosuite Stack cube identities (cubeA is red, cubeB is green)
CUBE_COLOR = {"cubeA": "red", "cubeB": "green"}
COLOR_CUBE = {v: k for k, v in CUBE_COLOR.items()}


class StackEnv:
    def __init__(self, camera: str = "agentview", img: int = 256, seed: Optional[int] = None):
        cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        self.camera = camera
        self.img = img
        self.env = suite.make(
            env_name="Stack", robots="Panda",
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=False, control_freq=20, controller_configs=cfg,
        )
        if seed is not None:
            np.random.seed(seed)
        self.obs = self.env.reset()
        self.frames: List[np.ndarray] = []

    # ---- observation ------------------------------------------------------
    @property
    def goal_text(self) -> str:
        return "stack the red cube on the green cube"

    def cube_pos(self, key: str) -> np.ndarray:
        return np.array(self.obs[f"{key}_pos"])

    def eef(self) -> np.ndarray:
        return np.array(self.obs["robot0_eef_pos"])

    def render(self) -> np.ndarray:
        return self.render_named(self.camera)

    def render_named(self, name: str) -> np.ndarray:
        rgb = self.env.sim.render(width=self.img, height=self.img, camera_name=name)
        return np.flipud(rgb).copy()

    def estimate_vision_poses(self, goal: str = "",
                              views=("agentview", "frontview", "sideview")
                              ) -> Dict[str, np.ndarray]:
        """Triangulate cube poses from MULTIPLE robosuite cameras via the multi-agent
        vision council — no privileged state. Returns {cube_key: world xyz}."""
        import os
        import tempfile

        import mujoco
        from PIL import Image

        from ..perception3d import camera as C
        from ..perception3d import estimator as E

        m, d = self.env.sim.model._model, self.env.sim.data._data
        mujoco.mj_forward(m, d)
        tmp = tempfile.mkdtemp(prefix="rs_vis_")
        pairs = []
        for name in views:
            p = os.path.join(tmp, f"{name}.png")
            Image.fromarray(self.render_named(name)).save(p)
            pairs.append((p, C.from_mujoco(m, d, name, self.img, self.img)))
        est = E.estimate_poses_multiview(pairs, goal=goal, n_agents=2,
                                         block_h=0.04, table_z=0.82)
        out: Dict[str, np.ndarray] = {}
        for pe in est.values():
            cube = COLOR_CUBE.get((pe.color or "").lower())
            if cube:
                out[cube] = pe.xyz
        return out

    def capture(self) -> None:
        self.frames.append(self.render())

    def check_success(self) -> bool:
        return bool(self.env._check_success())

    # ---- low-level control ------------------------------------------------
    def _move(self, target, grip: float, steps: int = 120, tol: float = 0.006,
              kp: float = 8.0, capture_every: int = 0) -> None:
        for i in range(steps):
            eef = self.eef()
            d = np.asarray(target) - eef
            a = np.zeros(7)
            a[:3] = np.clip(kp * d, -1, 1)
            a[6] = grip
            self.obs, _, _, _ = self.env.step(a)
            if capture_every and i % capture_every == 0:
                self.capture()
            if np.linalg.norm(d) < tol:
                break

    def _grip(self, g: float, n: int = 22, capture: bool = False) -> None:
        for _ in range(n):
            self.obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, float(g)]))
        if capture:
            self.capture()

    # ---- the skill the planner drives -------------------------------------
    def pick_place(self, src_cube: str, dst, capture_every: int = 8,
                   src_xyz=None, dst_xyz=None) -> None:
        """Pick src_cube and place it on dst (a cube key, or 'table').

        src_xyz/dst_xyz override the privileged cube poses — pass the vision-estimated
        poses here to grasp from perception instead of ground-truth state.
        """
        A = (np.asarray(src_xyz, float) if src_xyz is not None
             else self.cube_pos(src_cube)).copy()
        self._move(A + [0, 0, 0.12], OPEN, capture_every=capture_every)
        self._move(A + [0, 0, 0.0], OPEN, kp=5, capture_every=capture_every)
        self._grip(CLOSE, 25, capture=True)
        self._move(A + [0, 0, 0.20], CLOSE, capture_every=capture_every)
        if dst == "table":
            tgt = np.array([0.0, -0.15, 0.90])
            place_z = 0.84
        else:
            B = (np.asarray(dst_xyz, float) if dst_xyz is not None
                 else self.cube_pos(dst)).copy()
            tgt = B + [0, 0, 0.12]
            place_z = float(B[2]) + 0.055
        self._move(tgt, CLOSE, capture_every=capture_every)
        self._move([tgt[0], tgt[1], place_z], CLOSE, kp=5, capture_every=capture_every)
        self._grip(OPEN, 12, capture=True)
        self._move([tgt[0], tgt[1], place_z + 0.18], OPEN, capture_every=capture_every)

    def save_gif(self, path: str, fps: float = 12) -> Optional[str]:
        if not self.frames:
            return None
        from PIL import Image
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        return path

    def close(self) -> None:
        self.env.close()
