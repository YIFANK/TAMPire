"""Tier-1 physics verification of the long-horizon buried-base tower.

Plans the tower with the TAMPire pipeline (council, or `--baseline` myopic), then
EXECUTES the verified plan in the MuJoCo physics env and reads success back from
the simulated body positions. Two independent verdicts — the symbolic verifier
inside the pipeline and the physics verifier here — must agree. This proves the
long-horizon solve is real, not just symbolically valid.

    python -m tampire.simreal.longhorizon_run --n 5 --out runs/lh_tower
    python -m tampire.simreal.longhorizon_run --n 5 --baseline   # watch repairs
"""
from __future__ import annotations

import argparse
import os
from typing import List

from PIL import Image

from .. import llm, pipeline
from ..eval.longhorizon import solve_tower, tower_task
from ..schemas import Plan, Scene
from ..sim import feasibility
from .env import TabletopEnv


def run(n: int, *, baseline: bool, max_rounds: int, out_prefix: str,
        camera: str = "angled", save_frames: bool = True):
    from rich.console import Console
    console = Console()

    task = tower_task(n)
    console.print(f"[bold]Tier-1 long-horizon tower[/]  n={n}  "
                  f"start={task.scene.notes.split('start ')[1]}")

    # oracle reference (proves solvable) + verify physics start stack is stable
    oracle = solve_tower(task)
    ov = feasibility.check(task.scene, oracle, task.goal_predicates)
    console.print(f"[dim]oracle {len(oracle.steps)} steps  symbolic-feasible={ov.ok}[/]\n")

    # --- plan with the pipeline (council debates/repairs to a verified plan) ---
    llm.METRICS.reset()
    res = pipeline.run(
        task.goal,
        scene=Scene.from_dict(task.scene.to_dict()),
        goal_predicates=task.goal_predicates,
        myopic_planner=baseline,
        max_repair_rounds=max_rounds,
    )
    repairs = len(res.rounds) - 1
    plan = res.final_plan or Plan()
    console.print(f"planner={'baseline' if baseline else 'council'}  "
                  f"symbolic={'✓' if res.success else '✗'}  "
                  f"plan={len(plan.steps)} steps  repairs={repairs}  "
                  f"model={llm.METRICS.total_model_s:.2f}s")

    # --- execute the plan in PHYSICS, frame by frame ---
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    env = TabletopEnv(task.scene)
    frames: List[str] = []
    if save_frames:
        f0 = f"{out_prefix}_00.png"
        Image.fromarray(env.render(camera)).save(f0)
        frames.append(f0)

    fail_step = None
    for i, step in enumerate(plan.steps, 1):
        log = env.execute(Plan(steps=[step]))[0]
        if not log.ok and fail_step is None:
            fail_step = (i, str(step), log.note)
        if save_frames:
            f = f"{out_prefix}_{i:02d}.png"
            Image.fromarray(env.render(camera)).save(f)
            frames.append(f)

    phys_ok, phys_reason = env.check_goal(task.goal_predicates)

    gif = None
    if save_frames and len(frames) > 1:
        imgs = [Image.open(f).convert("RGB") for f in frames]
        gif = f"{out_prefix}.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=600, loop=0)

    # --- report ---
    agree = res.success == phys_ok
    console.print(
        f"\n[bold]symbolic verifier[/] : {'PASS' if res.success else 'FAIL'}"
        f"   [bold]physics verifier[/] : {'PASS' if phys_ok else 'FAIL'}"
        f"   verifiers agree: {'[green]yes[/]' if agree else '[red]NO[/]'}")
    console.print(f"[dim]physics: {phys_reason}[/]")
    if fail_step:
        console.print(f"[red]first exec failure: step {fail_step[0]} {fail_step[1]} -> {fail_step[2]}[/]")
    if gif:
        console.print(f"[dim]wrote {gif} ({len(frames)} frames)[/]")
    env.close()
    return res.success, phys_ok, agree


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-1 physics check of the long-horizon tower")
    ap.add_argument("--n", type=int, default=5, help="tower height (3..7)")
    ap.add_argument("--baseline", action="store_true", help="myopic planner (exercises repair)")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--out", default="runs/lh_tower", help="output prefix for frames + GIF")
    ap.add_argument("--camera", default="angled")
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    run(a.n, baseline=a.baseline, max_rounds=a.max_rounds, out_prefix=a.out,
        camera=a.camera, save_frames=not a.no_frames)


if __name__ == "__main__":
    main()
