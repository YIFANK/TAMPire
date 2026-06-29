"""Convenience: vision pose estimation directly from a TAMPire MuJoCo env, plus a
ready-made pose_provider for the skill controllers (Tier-2/Tier-3 style).

This is the non-privileged drop-in: instead of reading body positions from the sim,
poses come from the multi-agent vision council triangulating rendered camera views.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import numpy as np
from PIL import Image

from . import camera as C
from . import estimator as E
from .grounding3d import build_scene

_DEFAULT_VIEWS = ("angled", "angled_left", "angled_right")


def estimate_from_env(env, *, goal: str = "", views=_DEFAULT_VIEWS, n_agents: int = 2,
                      out_dir: str | None = None) -> Dict[str, E.PoseEstimate]:
    """Render `views` from a TabletopEnv-like object and triangulate object poses."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="tampire_p3d_")
    os.makedirs(out_dir, exist_ok=True)
    pairs: List = []
    for name in views:
        path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(env.render(name)).save(path)
        pairs.append((path, C.from_mujoco(env.model, env.data, name, 640, 480)))
    return E.estimate_poses_multiview(pairs, goal=goal, n_agents=n_agents)


def vision_scene_from_env(env, *, goal: str = "", **kw):
    """Full perceived Scene (objects + geometric predicates) for TAMP planning."""
    est = estimate_from_env(env, goal=goal, **kw)
    return build_scene(est, table_bounds=env.scene.table_bounds)


def pose_provider_from_env(env, *, goal: str = "", **kw):
    """Return a `{object_id: xyz}` map usable as a skill-controller pose_provider.
    Estimated ONCE from vision (objects don't move until the robot acts)."""
    est = estimate_from_env(env, goal=goal, **kw)
    poses = {oid: pe.xyz for oid, pe in est.items()}

    def provider(_obs=None, _env_id=0):
        return poses

    return provider, est
