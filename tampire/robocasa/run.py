"""Milestone A end-to-end: RoboCasa task -> perceive (from pixels + language) ->
TAMPire pipeline (council) -> VERIFIED long-horizon plan.

The headline: the instruction never says "open the cabinet", yet the verifier
rejects placing into a closed container and the council infers the open() step —
a genuine task-and-motion inference, zero-shot, from a recognized benchmark image.
"""
from __future__ import annotations

import argparse

from .. import llm, pipeline
from ..sim import feasibility
from . import perceive as P
from . import task as T


def run(task_name: str, *, seed: int, baseline: bool, max_rounds: int,
        camera: str = "robot0_agentview_left", execute: bool = False,
        real: bool = False) -> bool:
    execute = execute or real
    from rich.console import Console
    console = Console()

    # 1. real benchmark task -> image + NL instruction
    rc = T.load_task(task_name, seed=seed, camera=camera, keep_env=execute)
    console.print(f"[bold]RoboCasa task[/]: {task_name}  (seed {seed})")
    console.print(f"[bold cyan]instruction[/]: \"{rc.instruction}\"")
    console.print(f"[dim]frame: {rc.image_path}  ground-truth fixtures: "
                  f"{ {k:v for k,v in rc.truth.items() if k!='instruction'} }[/]\n")

    # 2. perceive from pixels (+instruction context)
    llm.METRICS.reset()
    pt = P.perceive(rc.image_path, rc.instruction)
    console.print(f"[bold]perceived[/]: object=[yellow]{pt.object_id}[/]  "
                  f"target=[yellow]{pt.target_id}[/]  "
                  f"target_closed={pt.target_closed}")
    console.print(f"[dim]notes: {pt.notes}[/]")
    console.print("[dim]initial predicates: "
                  + ", ".join(str(p) for p in pt.scene.predicates) + "[/]")
    console.print("[dim]goal predicates  : "
                  + ", ".join(str(p) for p in pt.goal_predicates) + "[/]\n")

    # 3. TAMPire pipeline -> verified plan (goal predicates injected; the council
    #    must still discover HOW, e.g. opening the closed cabinet)
    res = pipeline.run(
        rc.instruction,
        scene=pt.scene,
        goal_predicates=pt.goal_predicates,
        myopic_planner=baseline,
        max_repair_rounds=max_rounds,
    )

    plan = res.final_plan
    repairs = len(res.rounds) - 1
    inferred_open = bool(plan and any(s.action == "open" for s in plan.steps))
    console.print(f"[bold]plan[/] ({'baseline' if baseline else 'council'}, "
                  f"{'VERIFIED ✓' if res.success else 'unverified ✗'}, "
                  f"repairs={repairs}, model={llm.METRICS.total_model_s:.2f}s):")
    if plan:
        for i, s in enumerate(plan.steps, 1):
            tag = "  [green]<- inferred, not in the instruction[/]" if s.action == "open" else ""
            console.print(f"   {i}. {s}{tag}")

    # 4. independent re-check of the final plan
    v = feasibility.check(pt.scene, plan, pt.goal_predicates) if plan else None
    console.print(f"\n[bold]verifier[/]: {v.summary() if v else 'no plan'}")
    console.print(f"[bold]inferred the cabinet must be opened[/]: "
                  f"{'[green]YES[/]' if inferred_open else '[yellow]no[/]'}")

    # 5. (optional) EXECUTE the plan in the real RoboCasa env -> NATIVE success + video
    if execute and rc.env is not None and plan:
        from . import execute as X
        fx = X.target_fixture(rc.env, pt.target_id)
        mode = "REAL arm OSC" if real else "magic gripper"
        out = f"runs/rc_{'arm' if real else 'exec'}_{task_name}_{seed}"
        console.print(f"\n[bold]executing in RoboCasa[/] ({mode}) on fixture "
                      f"[yellow]{type(fx).__name__ if fx else None}[/] ...")
        if real:
            from . import arm_execute as R
            ex = R.execute_plan_real(rc.env, fx, plan, out_prefix=out, camera=camera)
            extra = f"  grasped={ex.grasped}"
        else:
            ex = X.execute_plan(rc.env, fx, plan, out_prefix=out, camera=camera)
            extra = ""
        console.print(f"[bold]RoboCasa NATIVE success[/]: "
                      f"{'[green]PASS ✓[/]' if ex.native_success else '[red]FAIL ✗[/]'}"
                      f"{extra}   steps: {ex.log}")
        if ex.gif:
            console.print(f"[dim]wrote {ex.gif} ({len(ex.frames)} frames)[/]")
        try:
            rc.env.close()
        except Exception:
            pass
        return bool(ex.native_success)
    return bool(res.success)


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboCasa Milestone A: perceive -> verified plan")
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--baseline", action="store_true", help="myopic planner (skips opening)")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--execute", action="store_true",
                    help="execute the plan in RoboCasa (magic gripper) -> native success + GIF")
    ap.add_argument("--real", action="store_true",
                    help="REAL closed-loop PandaOmron arm control (actual grasp) -> native success + GIF")
    a = ap.parse_args()
    ok = run(a.task, seed=a.seed, baseline=a.baseline, max_rounds=a.max_rounds,
             camera=a.camera, execute=a.execute, real=a.real)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
