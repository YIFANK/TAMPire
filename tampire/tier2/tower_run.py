"""Execute the long-horizon buried-base tower with a REAL Panda arm in robosuite.

Plans the tower with the TAMPire pipeline, then drives the Panda through the plan
with scripted OSC pick-place (actual grasps, real contact physics) — disassembling
the wrong column to table-parking spots and rebuilding it bottom-up. Success is
read geometrically from the simulated cube positions.

    python -m tampire.tier2.tower_run --n 3 --out runs/tower_arm
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

from .. import llm, pipeline
from ..eval.longhorizon import tower_task
from ..schemas import Scene

# free table spots to park disassembled cubes (xy on the table, around the column)
_PARK_SPOTS: List[Tuple[float, float]] = [
    (-0.17, -0.02), (0.17, -0.02), (-0.17, 0.15), (0.17, 0.15),
    (0.0, 0.17), (-0.17, -0.18), (0.17, -0.18),
]


def _color(block_id: str) -> str:
    return block_id.replace("_block", "")


def run(n: int, *, baseline: bool, max_rounds: int, out_prefix: str,
        camera: str = "frontview", seed: int = 0):
    from rich.console import Console
    from robosuite.controllers import load_composite_controller_config
    from .tower_env import TowerEnv

    console = Console()
    task = tower_task(n)
    console.print(f"[bold]Real-arm long-horizon tower[/]  n={n}")
    console.print(f"[dim]start column (bottom->top): {task.start_order}[/]")
    console.print(f"[dim]goal tower   (bottom->top): {task.goal_order}[/]\n")

    # 1. plan with the council
    llm.METRICS.reset()
    res = pipeline.run(task.goal, scene=Scene.from_dict(task.scene.to_dict()),
                       goal_predicates=task.goal_predicates,
                       myopic_planner=baseline, max_repair_rounds=max_rounds)
    plan = res.final_plan
    repairs = len(res.rounds) - 1
    console.print(f"plan ({'baseline' if baseline else 'council'}, "
                  f"{'VERIFIED ✓' if res.success else '✗'}, repairs={repairs}, "
                  f"{len(plan.steps) if plan else 0} steps, model={llm.METRICS.total_model_s:.2f}s)")
    if not plan:
        return False

    # 2. build the real-Panda env with the buried-base start column
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = TowerEnv(start_order=task.start_order, column_xy=(0.0, 0.0),
                   robots="Panda", controller_configs=cfg, has_renderer=False,
                   has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
                   ignore_done=True, horizon=100000)
    env.boot(seed=seed)
    for _ in range(40):                       # settle the start column
        env.obs, _, _, _ = env.step(np.zeros(7))
    env.capture(camera)

    # 3. execute: each `place(obj, target)` drives a full scripted pick-place
    park_i = 0
    for s in plan.steps:
        if s.action != "place" or len(s.args) < 2:
            continue
        obj, target = _color(s.args[0]), s.args[1]
        if target == "table":
            spot = _PARK_SPOTS[park_i % len(_PARK_SPOTS)]
            park_i += 1
            console.print(f"  pick [yellow]{obj}[/] -> park at {spot}")
            env.pick_place(obj, "table", park_xy=spot, cam=camera)
        else:
            console.print(f"  pick [yellow]{obj}[/] -> stack on [yellow]{_color(target)}[/]")
            env.pick_place(obj, _color(target), cam=camera)

    # 4. geometric success from the real cube positions
    ok = True
    h = env._cube_half
    details = []
    for gp in task.goal_predicates:
        a, b = _color(gp.args[0]), _color(gp.args[1])
        pa, pb = env.cube_pos(a), env.cube_pos(b)
        aligned = abs(pa[0] - pb[0]) < 0.03 and abs(pa[1] - pb[1]) < 0.03
        stacked = abs(pa[2] - (pb[2] + 2 * h)) < 0.02
        good = aligned and stacked
        ok = ok and good
        details.append(f"on({a},{b}): {'✓' if good else '✗'} dz={pa[2]-pb[2]:+.3f}")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    gif = env.save_gif(f"{out_prefix}.gif")
    console.print(f"\n[bold]real-arm tower success[/]: "
                  f"{'[green]PASS ✓[/]' if ok else '[red]FAIL ✗[/]'}")
    for d in details:
        console.print(f"   {d}")
    if gif:
        console.print(f"[dim]wrote {gif} ({len(env.frames)} frames)[/]")
    env.close()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Real Panda arm executes the long-horizon tower")
    ap.add_argument("--n", type=int, default=3, help="tower height (3..7; start small)")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--out", default="runs/tower_arm")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ok = run(a.n, baseline=a.baseline, max_rounds=a.max_rounds, out_prefix=a.out,
             camera=a.camera, seed=a.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
