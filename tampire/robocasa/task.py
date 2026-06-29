"""Wrap a RoboCasa env into the inputs TAMPire needs: a rendered camera frame and
the task's natural-language instruction, plus the ground-truth roles (for honest
reporting / id alignment). robocasa/mujoco imports are deferred to call time."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RoboCasaTask:
    task_name: str
    instruction: str                       # env's NL `lang`
    image_path: str                        # rendered camera frame for perception
    camera: str
    truth: Dict[str, str] = field(default_factory=dict)  # ground-truth roles
    env: object = None                     # the live env (for Milestone B / execution)


def load_task(task_name: str = "PickPlaceCounterToCabinet", *, seed: int = 3,
              camera: str = "robot0_agentview_left", width: int = 640, height: int = 480,
              out_dir: str = "runs", keep_env: bool = False) -> RoboCasaTask:
    os.environ.setdefault("MUJOCO_GL", "cgl")   # offscreen GL on macOS
    from PIL import Image
    import robocasa  # noqa: F401  (registers envs)
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name=task_name, render_onscreen=False, seed=seed)
    env.reset()
    meta = env.get_ep_meta()
    instruction = meta.get("lang", "").strip()

    img = env.sim.render(width=width, height=height, camera_name=camera)[::-1]  # flip vertical
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, f"rc_{task_name}_{seed}.png")
    Image.fromarray(np.asarray(img)).save(image_path)

    # ground-truth roles from ep_meta (for reporting, not fed to perception)
    truth: Dict[str, str] = {}
    refs = meta.get("fixture_refs", {}) or {}
    for role, fx in refs.items():
        truth[role] = str(fx).split("_")[0]   # e.g. hingecabinet_2_... -> hingecabinet
    truth["instruction"] = instruction

    if not keep_env:
        try:
            env.close()
        except Exception:
            pass
        env = None
    return RoboCasaTask(task_name=task_name, instruction=instruction,
                        image_path=image_path, camera=camera, truth=truth, env=env)
