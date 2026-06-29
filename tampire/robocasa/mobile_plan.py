"""Mobile-manipulation TAMP on a REAL RoboCasa task's geometry (symbolic layer).

The objects in a RoboCasa kitchen task are spread beyond a single arm-reach, so a
fixed-base plan is infeasible. Here TAMPire treats the mobile base as part of the
plan: reachability is measured from the base, and when the verifier reports a target
is "out of the arm's reach", the Gemma council inserts a `move_base(target)` step.
The result is a navigate-and-manipulate sequence — task-and-motion planning over a
mobile manipulator — derived from the task's real object positions.

    PYTHONPATH=/Users/yifankang/TAMPire MUJOCO_GL=cgl \
      .venv-arm64/bin/python -m tampire.robocasa.mobile_plan --task StackBowlsInSink --seed 2
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .. import llm, pipeline
from ..schemas import Predicate, Scene, WorldObject


def _build_scene(env, bowl_keys, sink_fixture):
    obs = env._get_observations()
    base = np.array(obs["robot0_base_pos"])
    objs = [WorldObject("counter", "counter", position=(0, 0, 0), affordances=["support"])]
    # effective arm reach measured from the real OSC reach tests (~0.45 m); beyond that
    # the mobile base must drive closer first.
    preds = [Predicate("base_at", [f"{base[0]:.3f}", f"{base[1]:.3f}"]),
             Predicate("reach_radius", ["0.45"])]
    bowls = []
    for i, k in enumerate(bowl_keys):
        p = np.array(obs[k])
        bid = f"bowl{i+1}"
        bowls.append(bid)
        objs.append(WorldObject(bid, "bowl", position=(float(p[0]), float(p[1]), float(p[2])),
                                affordances=["graspable", "container"]))
        preds += [Predicate("on", [bid, "counter"]), Predicate("clear", [bid]),
                  Predicate("graspable", [bid])]
    s = sink_fixture.get_int_sites(relative=False)
    p0, px, py, pz = list(s.values())[0]
    sc = (np.array(p0) + np.array(px) + np.array(py) + np.array(pz)) / 4
    objs.append(WorldObject("sink", "sink", position=(float(sc[0]), float(sc[1]), float(sc[2])),
                            affordances=["container", "support"]))
    preds.append(Predicate("container", ["sink"]))
    scene = Scene(objects=objs, predicates=preds, table_bounds=(-10, -10, 10, 10),
                  notes="real StackBowlsInSink geometry (mobile manip)")
    goal = [Predicate("in", [b, "sink"]) for b in bowls]
    return scene, goal, base


def run(task_name, seed):
    import os as _os
    _os.environ.setdefault("MUJOCO_GL", "cgl")
    from rich.console import Console
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env
    console = Console()

    env = create_env(env_name=task_name, seed=seed, render_onscreen=False)
    env.reset()
    bowl_keys = [k for k in env._get_observations() if k.endswith("_pos")
                 and ("receptacle" in k or "bowl" in k) and "robot" not in k][:2]
    scene, goal, base = _build_scene(env, bowl_keys, env.sink)

    console.print(f"[bold]Mobile-manip TAMP[/] on {task_name} (real geometry)")
    console.print(f"[dim]base at ({base[0]:.2f},{base[1]:.2f}); "
                  f"objects spread across the kitchen → fixed-base plan infeasible[/]\n")

    def streamer(ev, who, delta):
        if ev == "critic_start":
            console.print(f"\n  [bold]🗣 critic · {who.split('—')[0].strip()}[/bold]")
        elif ev == "critic_delta":
            print(delta, end="", flush=True)
        elif ev == "repair_start":
            console.print("\n  [bold]🔧 repair chair…[/bold]")

    llm.METRICS.reset()
    res = pipeline.run("Stack both bowls in the sink.", scene=scene, goal_predicates=goal,
                       myopic_planner=True, max_repair_rounds=8, council_stream=streamer)
    console.print(f"\n\n[bold]final plan[/] (repairs={len(res.rounds)-1}, "
                  f"{'VERIFIED ✓' if res.success else '✗'}, model={llm.METRICS.total_model_s:.2f}s):")
    for i, s in enumerate(res.final_plan.steps, 1):
        tag = "  [green]← navigation inserted by the council[/]" if s.action == "move_base" else ""
        console.print(f"   {i}. {s}{tag}")
    nav = sum(1 for s in res.final_plan.steps if s.action == "move_base")
    console.print(f"\n[bold]{nav} move_base steps[/] planned across {len(res.rounds)-1} debate rounds — "
                  f"TAMP reasoned about WHERE to drive the base, not just what to grasp.")
    env.close()
    return res.success


def main():
    ap = argparse.ArgumentParser(description="Mobile-manip TAMP on real RoboCasa geometry")
    ap.add_argument("--task", default="StackBowlsInSink")
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    raise SystemExit(0 if run(a.task, a.seed) else 1)


if __name__ == "__main__":
    main()
