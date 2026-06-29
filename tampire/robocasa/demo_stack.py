"""MAIN DEMO — StackBowlsCabinet: the full TAMPire story in one RoboCasa task.

  multimodal  : a Gemma-4 vision agent reads the two bowls + open cabinet from pixels
  multi-agent : the myopic plan nests the small bowl first -> the large bowl is no
                longer clear -> the 3-critic council debates and REORDERS the plan
  long-horizon: a multi-stage stack-and-store (place large in cabinet, nest small in it)
  speed       : the whole debate is a handful of Gemma calls on Cerebras (~sub-second compute)
  robotics    : executed in real RoboCasa physics, scored by its NATIVE success check

Run (in the RoboCasa venv):
  PYTHONPATH=/Users/yifankang/TAMPire MUJOCO_GL=cgl \
    .venv-arm64/bin/python -m tampire.robocasa.demo_stack --seed 2 --baseline
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from .. import llm, pipeline
from ..schemas import Predicate, Scene, WorldObject
from ..sim import feasibility
from . import task as T
from .execute import _interior_centroid

STACK_SYS = """You are the PERCEPTION agent of a kitchen robot. The image shows a counter with
TWO bowls and an open cabinet above/near the counter. Identify the two bowls and decide which
is LARGER, and confirm the cabinet they should go into. Return ONLY JSON:
{"bowls": [{"size": "large", "color": "<word or null>"},
           {"size": "small", "color": "<word or null>"}],
 "target": {"name": "cabinet"},
 "notes": "one short sentence about what you see"}"""

STACK_USER = """Task instruction: "{instruction}"
Identify the larger and smaller bowl on the counter and the target cabinet."""


def perceive_stack(image_path: str, instruction: str):
    msgs = [llm.sys(STACK_SYS),
            llm.user_with_image(STACK_USER.format(instruction=instruction), image_path)]
    data, _ = llm.chat_json(msgs, label="rc_stack_perception", temperature=0.1)
    notes = str(data.get("notes", ""))

    # symbolic scene: two bowls (containers) on the counter + an open cabinet
    objs = [
        WorldObject("large_bowl", "bowl", position=(-0.1, -0.20, 0.05),
                    size=(0.16, 0.16, 0.07), affordances=["graspable", "container", "support"]),
        WorldObject("small_bowl", "bowl", position=(0.1, -0.20, 0.05),
                    size=(0.11, 0.11, 0.05), affordances=["graspable", "container", "support"]),
        WorldObject("counter", "counter", position=(0.0, -0.20, 0.0),
                    size=(0.6, 0.3, 0.02), affordances=["support"]),
        WorldObject("cabinet", "cabinet", position=(0.0, -0.10, 0.30),
                    size=(0.4, 0.4, 0.4), affordances=["container", "openable"]),
    ]
    preds = [
        Predicate("on", ["large_bowl", "counter"]), Predicate("graspable", ["large_bowl"]),
        Predicate("clear", ["large_bowl"]),
        Predicate("on", ["small_bowl", "counter"]), Predicate("graspable", ["small_bowl"]),
        Predicate("clear", ["small_bowl"]),
        Predicate("container", ["large_bowl"]), Predicate("container", ["cabinet"]),
        Predicate("open", ["cabinet"]),
    ]
    scene = Scene(objects=objs, predicates=preds, notes=f"robocasa stack: {notes}")
    # goal: smaller bowl nested in larger, larger bowl stored in the cabinet.
    # (nesting listed first: a greedy/myopic planner nests on the counter first, which
    # makes the large bowl un-clear and un-pickable — the long-horizon trap the council
    # must catch and REORDER. A precondition-aware planner stores the large bowl first.)
    goal = [Predicate("in", ["small_bowl", "large_bowl"]),
            Predicate("in", ["large_bowl", "cabinet"])]
    return scene, goal, notes


# --- animated magic execution of the stack in RoboCasa ---
def _set(sim, joint, xyz):
    sim.data.set_joint_qpos(joint, np.concatenate([np.asarray(xyz, float), [1, 0, 0, 0]]))
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def _glide(env, joint, a, b, frames, snap):
    a, b = np.asarray(a, float), np.asarray(b, float)
    for t in np.linspace(0, 1, frames):
        _set(env.sim, joint, a + (b - a) * t)
        snap()


def execute_stack(env, plan, out_prefix, camera="robot0_agentview_left"):
    """Drive the plan: place large bowl in cabinet, nest small bowl in it. The symbolic
    ids (large_bowl/small_bowl) map to RoboCasa bodies bowl2(large)/bowl1(small)."""
    from PIL import Image
    frames: List[str] = []
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    idx = [0]

    def snap():
        img = env.sim.render(width=640, height=480, camera_name=camera)[::-1]
        p = f"{out_prefix}_{idx[0]:03d}.png"
        Image.fromarray(np.asarray(img)).save(p)
        frames.append(p)
        idx[0] += 1

    cab = _interior_centroid(env.cabinet)
    BODY = {"large_bowl": "bowl2_joint0", "small_bowl": "bowl1_joint0"}
    snap()
    for s in plan.steps:
        if s.action != "place" or len(s.args) < 2:
            continue
        obj, target = s.args[0], s.args[1]
        if obj not in BODY:
            continue
        start = np.array(env.sim.data.get_joint_qpos(BODY[obj])[:3])
        if target == "cabinet":
            dest = cab
        elif target == "large_bowl":
            dest = np.array(env.sim.data.get_joint_qpos(BODY["large_bowl"])[:3]) + [0, 0, 0.04]
        else:
            continue
        lift = max(start[2], dest[2]) + 0.12
        _glide(env, BODY[obj], start, [start[0], start[1], lift], 8, snap)
        _glide(env, BODY[obj], [start[0], start[1], lift], [dest[0], dest[1], lift], 12, snap)
        _glide(env, BODY[obj], [dest[0], dest[1], lift], dest, 8, snap)
    for _ in range(60):
        env.sim.step()
    snap()
    native = bool(env._check_success())
    gif = f"{out_prefix}.gif"
    imgs = [Image.open(f).convert("RGB") for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=110, loop=0)
    return native, gif, frames


def run(seed: int, baseline: bool, max_rounds: int, camera: str):
    from rich.console import Console
    console = Console()

    rc = T.load_task("StackBowlsCabinet", seed=seed, camera=camera, keep_env=True)
    console.print(f"[bold]MAIN DEMO · StackBowlsCabinet[/]  (seed {seed})")
    console.print(f"[bold cyan]instruction[/]: \"{rc.instruction}\"\n")

    llm.METRICS.reset()
    scene, goal, notes = perceive_stack(rc.image_path, rc.instruction)
    console.print(f"[bold]① multimodal perception[/] (Gemma vision): {notes}")
    console.print(f"[dim]goal: {', '.join(str(p) for p in goal)}[/]\n")

    # stream the council debate
    def streamer(event, who, delta):
        if event == "critic_start":
            console.print(f"\n  [bold]🗣  critic · {who}[/bold]")
        elif event == "critic_delta":
            print(delta, end="", flush=True)
        elif event == "repair_start":
            console.print("\n\n  [bold]🔧 repair chair synthesizes a fix…[/bold]")

    console.print("[bold]②–③ planning + multi-agent debate[/] (Gemma council on Cerebras):")
    res = pipeline.run(rc.instruction, scene=scene, goal_predicates=goal,
                       myopic_planner=baseline, max_repair_rounds=max_rounds,
                       council_stream=streamer)
    plan = res.final_plan
    console.print(f"\n\n[bold]verified plan[/] (repairs={len(res.rounds)-1}, "
                  f"model={llm.METRICS.total_model_s:.2f}s, "
                  f"{llm.METRICS.tokens_per_s:.0f} tok/s):")
    for i, s in enumerate(plan.steps, 1):
        console.print(f"   {i}. {s}")
    v = feasibility.check(scene, plan, goal)
    console.print(f"[dim]symbolic verifier: {v.summary()}[/]")

    console.print("\n[bold]④ executing in RoboCasa[/] -> native success + video …")
    out = f"runs/rc_stack_{seed}"
    native, gif, frames = execute_stack(rc.env, plan, out, camera=camera)
    console.print(f"[bold]RoboCasa NATIVE success[/]: "
                  f"{'[green]PASS ✓[/]' if native else '[red]FAIL ✗[/]'}")
    console.print(f"[dim]wrote {gif} ({len(frames)} frames)[/]")
    try:
        rc.env.close()
    except Exception:
        pass
    return native


def main():
    ap = argparse.ArgumentParser(description="TAMPire MAIN DEMO: StackBowlsCabinet")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--baseline", action="store_true", help="myopic plan -> triggers council debate")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--camera", default="robot0_agentview_left")
    a = ap.parse_args()
    ok = run(a.seed, a.baseline, a.max_rounds, a.camera)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
