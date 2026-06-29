"""Validate the full VISION-BASED TAMP loop and quantify it.

For each generated scene:
  render -> multi-agent vision pose estimate -> geometric grounding -> Scene
  -> TAMP (plan + motion-feasibility) -> execute in physics -> success.

Metrics: pose error (cm) vs ground truth, stack-detection accuracy, and physics
success of the plan TAMP produced from PERCEIVED geometry (no privileged state).

  python -m tampire.perception3d.bench --n 6
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from .. import pipeline
from ..eval import scenegen
from ..llm import METRICS
from ..simreal.env import TabletopEnv
from . import camera as C
from . import estimator as E
from . import grounding3d

console = Console()


@dataclass
class Row:
    seed: int
    pose_err_cm: float
    n_matched: int
    stack_ok: bool
    plan_physics_ok: bool


def _gt_on(scene) -> set:
    return {(p.args[0], p.args[1]) for p in scene.predicates
            if p.name == "on" and len(p.args) == 2 and p.args[1] != "table"}


def run_one(seed: int, *, camera: str, n_agents: int, multiview: bool,
            depth: bool = False) -> Optional[Row]:
    task = scenegen.generate(seed)
    env = TabletopEnv(task.scene)
    if depth:
        from . import perceive as P
        ps = P.perceive(env, task.goal, n_agents=n_agents)
        vscene = ps.scene
        # pose error vs ground truth (match by color)
        gt = {o.id: np.array(o.position) for o in task.scene.objects}
        errs = [float(np.linalg.norm(ps.geoms[c].xy - gt[oid][:2]) * 100)
                for c in ps.geoms for oid in [next((k for k in gt if c in k), None)] if oid]
        pose_err = float(np.mean(errs)) if errs else float("nan")
        stack_ok = _gt_on(task.scene) == _gt_on(vscene)
        plan_ok = False
        try:
            res = pipeline.run(task.goal, scene=vscene, goal_predicates=task.goal_predicates)
            if res.final_plan:
                env2 = TabletopEnv(task.scene)
                env2.execute(res.final_plan)
                plan_ok, _ = env2.check_goal(task.goal_predicates)
                env2.close()
        except Exception as e:
            console.print(f"  [yellow]seed {seed} plan/exec error: {e}[/yellow]")
        env.close()
        return Row(seed, pose_err, len(errs), stack_ok, bool(plan_ok))
    if multiview:
        cams = ["angled", "angled_left", "angled_right"]
        views = []
        for cname in cams:
            ip = f"runs/p3d_{seed}_{cname}.png"
            Image.fromarray(env.render(cname)).save(ip)
            views.append((ip, C.from_mujoco(env.model, env.data, cname, 640, 480)))
        est = E.estimate_poses_multiview(views, goal=task.goal, n_agents=n_agents)
    else:
        img = f"runs/p3d_{seed}.png"
        Image.fromarray(env.render(camera)).save(img)
        cam = C.from_mujoco(env.model, env.data, camera, 640, 480)
        est = E.estimate_poses(img, cam, goal=task.goal, n_agents=n_agents)

    # pose error vs ground truth (match by color)
    gt = {o.id: np.array(o.position) for o in task.scene.objects}
    errs: List[float] = []
    for oid, pe in est.items():
        m = next((k for k in gt if pe.color and pe.color in k), None)
        if m:
            errs.append(float(np.linalg.norm(pe.xyz[:2] - gt[m][:2]) * 100))
    pose_err = float(np.mean(errs)) if errs else float("nan")

    # vision-grounded scene -> stack detection check
    vscene = grounding3d.build_scene(est, table_bounds=task.scene.table_bounds)
    stack_ok = _gt_on(task.scene) == _gt_on(vscene)

    # TAMP plan from PERCEIVED geometry, executed in real physics
    plan_ok = False
    try:
        res = pipeline.run(task.goal, scene=vscene, goal_predicates=task.goal_predicates)
        if res.final_plan:
            env2 = TabletopEnv(task.scene)
            env2.execute(res.final_plan)
            plan_ok, _ = env2.check_goal(task.goal_predicates)
            env2.close()
    except Exception as e:
        console.print(f"  [yellow]seed {seed} plan/exec error: {e}[/yellow]")
    env.close()
    return Row(seed, pose_err, len(errs), stack_ok, bool(plan_ok))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vision-based TAMP validation")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--camera", default="angled", choices=["angled", "topdown"])
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--multiview", action="store_true",
                    help="fuse a council across 3 angled camera views (occlusion-robust)")
    ap.add_argument("--depth", action="store_true",
                    help="TiPToP-style: semantic council + multi-view DEPTH geometry")
    args = ap.parse_args(argv)

    mode = ("depth + semantic council" if args.depth
            else "3-view council" if args.multiview else f"single {args.camera} view")
    console.print(f"[bold]Vision-based TAMP[/bold]  {args.n} scenes  ·  {args.agents} agents/view "
                  f"·  {mode}\n[dim]no privileged state; poses estimated from pixels[/dim]\n")

    rows: List[Row] = []
    for i in range(args.n):
        METRICS.reset()
        try:
            r = run_one(args.seed + i, camera=args.camera, n_agents=args.agents,
                        multiview=args.multiview, depth=args.depth)
        except Exception as e:
            console.print(f"  [yellow]seed {args.seed+i} skipped: {str(e)[:80]}[/yellow]")
            continue
        if r:
            rows.append(r)
            pe = "[green]✓[/green]" if r.plan_physics_ok else "[red]✗[/red]"
            sk = "✓" if r.stack_ok else "✗"
            console.print(f"  seed {r.seed}  pose_err={r.pose_err_cm:4.1f}cm  "
                          f"stack {sk}  plan→physics {pe}", highlight=False)

    valid = [r for r in rows if not np.isnan(r.pose_err_cm)]
    t = Table(title="\nVision-based TAMP — summary", title_style="bold")
    for c in ("metric", "value"):
        t.add_column(c)
    t.add_row("scenes", str(len(rows)))
    t.add_row("mean pose error", f"{np.mean([r.pose_err_cm for r in valid]):.1f} cm")
    t.add_row("stack-detection acc", f"{np.mean([r.stack_ok for r in rows])*100:.0f}%")
    t.add_row("plan→physics success", f"{np.mean([r.plan_physics_ok for r in rows])*100:.0f}%")
    console.print(t)
    console.print("[dim]plan→physics: TAMP plan built ONLY from perceived geometry, "
                  "executed and scored in physics.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
