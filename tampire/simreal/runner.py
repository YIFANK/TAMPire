"""Close the full loop through the real sim:

  MuJoCo scene -> render REAL pixels -> TAMPire pipeline (perception ... plan,
  debate-repair) -> execute the plan back in MuJoCo -> success read from PHYSICS.

Two independent verdicts are reported: the symbolic verifier's (inside the
pipeline) and the physics verifier's (this env). They should agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from .. import pipeline
from ..schemas import Plan, PlanStep, Predicate, Scene
from .env import TabletopEnv


@dataclass
class RealRunResult:
    goal: str
    symbolic_success: bool
    physics_success: bool
    physics_reason: str
    final_plan: Optional[Plan]
    repairs: int
    frames: List[str]
    gif: Optional[str]
    init_image: str


def _align_ids(perceived: Scene, env_scene: Scene) -> Dict[str, str]:
    """Map perceived object ids -> ground-truth env body ids by (color, type)."""
    out: Dict[str, str] = {}
    used = set()
    env_objs = list(env_scene.objects)
    for po in perceived.objects:
        p_is_container = po.category in ("bowl", "cup", "plate", "tray") or "container" in po.affordances
        best = None
        for eo in env_objs:
            if eo.id in used:
                continue
            e_is_container = eo.category in ("bowl", "cup", "plate", "tray") or "container" in eo.affordances
            if e_is_container != p_is_container:
                continue
            if (po.color or "").lower() == (eo.color or "").lower():
                best = eo.id
                break
        if best:
            out[po.id] = best
            used.add(best)
    return out


def _remap(plan: Plan, id_map: Dict[str, str]) -> Plan:
    steps = [PlanStep(s.action, [id_map.get(a, a) for a in s.args], s.rationale)
             for s in plan.steps]
    return Plan(steps=steps, rationale=plan.rationale)


def run_from_pixels(
    env_scene: Scene, goal: str, goal_predicates: List[Predicate],
    *, camera: str = "angled", baseline: bool = False, out_prefix: str = "runs/mjrun",
    events=None, save_frames: bool = True, max_repair_rounds=None,
) -> RealRunResult:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    env = TabletopEnv(env_scene)

    # 1. render REAL pixels and save for the perception agent (always needed)
    init_path = f"{out_prefix}_initial.png"
    Image.fromarray(env.render(camera)).save(init_path)

    # 2. full pipeline FROM the rendered image
    res = pipeline.run(goal, image_path=init_path, myopic_planner=baseline, events=events,
                       max_repair_rounds=max_repair_rounds)

    # 3. align perceived ids -> env body ids, execute in physics
    id_map = _align_ids(res.scene, env_scene)
    exec_plan = _remap(res.final_plan, id_map) if res.final_plan else Plan()
    frames = [init_path]
    if exec_plan.steps:
        for i, step in enumerate(exec_plan.steps, 1):
            env.execute(Plan(steps=[step]))
            if save_frames:
                f = f"{out_prefix}_{i:02d}.png"
                Image.fromarray(env.render(camera)).save(f)
                frames.append(f)

    # 4. physics-grounded success on the TRUE goal predicates
    phys_ok, phys_reason = env.check_goal(goal_predicates)

    gif = None
    if save_frames:
        final_path = f"{out_prefix}_final.png"
        Image.fromarray(env.render(camera)).save(final_path)
        frames.append(final_path)
        if len(frames) > 1:
            imgs = [Image.open(f).convert("RGB") for f in frames]
            gif = f"{out_prefix}.gif"
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=900, loop=0)

    env.close()
    return RealRunResult(
        goal=goal,
        symbolic_success=res.success,
        physics_success=phys_ok,
        physics_reason=phys_reason,
        final_plan=exec_plan,
        repairs=len(res.rounds) - 1,
        frames=frames, gif=gif, init_image=init_path,
    )
