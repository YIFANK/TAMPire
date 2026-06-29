"""Drive robosuite's Stack task with the full TAMPire pipeline, from real pixels.

  render agentview -> perception -> grounding -> goalspec -> plan -> (repair)
  -> map planned primitives to scripted pick-place skills -> execute on the Panda
  -> robosuite's native success criterion.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from .. import pipeline
from ..schemas import Plan, Scene
from .rs_env import COLOR_CUBE, StackEnv

_COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "cyan")


@dataclass
class Tier2Result:
    goal: str
    plan: Optional[Plan]
    executed_skills: List[str]
    symbolic_success: bool
    robosuite_success: bool
    repairs: int
    init_image: str
    gif: Optional[str]
    vision: bool = False
    vision_pose_error_cm: Optional[float] = None


def _color_of(scene: Scene, oid: str) -> Optional[str]:
    o = scene.by_id(oid)
    if o and o.color:
        c = o.color.lower()
        if c in _COLOR_WORDS:
            return c
    m = re.search("|".join(_COLOR_WORDS), oid.lower())
    return m.group(0) if m else None


def _to_cube(scene: Scene, oid: str) -> Optional[str]:
    if oid == "table":
        return "table"
    col = _color_of(scene, oid)
    return COLOR_CUBE.get(col) if col else None


def run_stack(*, seed: int = 0, baseline: bool = False, vision: bool = False,
              out_prefix: str = "runs/rs", events=None) -> Tier2Result:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    env = StackEnv(seed=seed)

    # 1. real agentview pixels for the perception agent
    init_path = f"{out_prefix}_initial.png"
    Image.fromarray(env.render()).save(init_path)
    env.capture()

    # 2. full pipeline from the rendered image
    res = pipeline.run(env.goal_text, image_path=init_path,
                       myopic_planner=baseline, events=events)

    # 2b. optionally estimate cube poses from vision (no privileged state)
    vposes = env.estimate_vision_poses(env.goal_text) if vision else {}

    # 3. map planned place() ops to scripted pick-place skills
    skills: List[str] = []
    if res.final_plan:
        for s in res.final_plan.steps:
            if s.action == "place" and len(s.args) >= 2:
                src = _to_cube(res.scene, s.args[0])
                dst = _to_cube(res.scene, s.args[1])
                if src and dst and src != "table":
                    env.pick_place(src, dst,
                                   src_xyz=vposes.get(src), dst_xyz=vposes.get(dst))
                    tag = "vision" if vision else "privileged"
                    skills.append(f"pick_place({src} -> {dst}) [{tag} poses]")

    rs_ok = env.check_success()
    Image.fromarray(env.render()).save(f"{out_prefix}_final.png")
    env.capture()
    gif = env.save_gif(f"{out_prefix}.gif")
    vis_err = None
    if vision and vposes:
        errs = [float(np.linalg.norm(np.asarray(vposes[k]) - env.cube_pos(k)) * 100)
                for k in vposes if k in ("cubeA", "cubeB")]
        vis_err = float(np.mean(errs)) if errs else None
    env.close()

    return Tier2Result(
        goal="stack the red cube on the green cube",
        plan=res.final_plan, executed_skills=skills,
        symbolic_success=res.success, robosuite_success=rs_ok,
        repairs=len(res.rounds) - 1, init_image=init_path, gif=gif,
        vision=vision, vision_pose_error_cm=vis_err,
    )
