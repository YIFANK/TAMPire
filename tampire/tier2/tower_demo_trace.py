"""Record the long-horizon TOWER end-to-end for the dashboard:

  PLAN  : Gemma multi-agent council plans the buried-base tower (myopic plan fails →
          critics → repair → verified). Captures rounds, critics, timing/throughput.
  EXECUTE: a REAL Panda arm runs the plan with scripted OSC pick-place. Captures the
          render frames, the frame range of every action, and — checked geometrically
          from the real cube positions after each action — which goal predicates hold.

The whole point: a 16-step long-horizon plan is synthesized in a fraction of a second
because Gemma-4 runs on Cerebras, and then a real arm carries it out.

    python -m tampire.tier2.tower_demo_trace --n 5
"""
from __future__ import annotations

import argparse
import base64
import io
import json

import numpy as np

OUT = "/Users/yifankang/TAMPire/runs/trace_tower.json"


def _pred_status_sym(scene, plan, goal_preds):
    from ..sim.world import World
    w = World.from_scene(scene)
    for s in plan.steps:
        ok, _ = w.apply(s)
        if not ok:
            break
    return [bool(w.satisfies([p])) for p in goal_preds]


def _mk_round(idx, plan, verdict, critics, scene, goal):
    failed = next((c for c in verdict.checks if not c.ok), None)
    return {
        "round": idx,
        "steps": [{"action": s.action, "args": s.args, "move_base": False} for s in plan.steps],
        "verdict_ok": bool(verdict.ok and verdict.goal_satisfied),
        "failed_step": (failed.index + 1) if failed else None,
        "failed_reason": failed.reason if failed else None,
        "critics": critics,
        "pred_satisfied": _pred_status_sym(scene, plan, goal),
    }


def _jpeg_b64(arr, width=300, q=70):
    from PIL import Image
    im = Image.fromarray(arr)
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def run(n, seed=0, max_rounds=8, camera="frontview"):
    from rich.console import Console
    from robosuite.controllers import load_composite_controller_config
    from .. import llm, pipeline
    from ..eval.longhorizon import tower_task
    from ..schemas import Scene
    from .tower_env import TowerEnv
    from .tower_run import _PARK_SPOTS, _color
    console = Console()

    task = tower_task(n)
    scene = Scene.from_dict(task.scene.to_dict())
    goal = task.goal_predicates

    # ---- PLAN (capture council trace) ----
    rounds = []
    cur = {"critics": []}
    state = {"buf": [], "persona": None}

    def stream(ev, who, delta):
        if ev == "critic_start":
            state["persona"] = who.split("—")[0].strip(); state["buf"] = []
        elif ev == "critic_delta":
            state["buf"].append(delta)
        elif ev == "critic_end":
            cur["critics"].append({"persona": state["persona"], "text": "".join(state["buf"]).strip()})

    def events(stage, payload):
        if stage == "plan_done":
            rounds.append(_mk_round(0, payload["plan"], payload["verdict"], [], scene, goal))
        elif stage == "debate_start":
            cur["critics"] = []
        elif stage == "repair_done":
            rounds.append(_mk_round(payload["round"], payload["plan"], payload["verdict"],
                                    list(cur["critics"]), scene, goal))

    llm.METRICS.reset()
    res = pipeline.run(task.goal, scene=Scene.from_dict(task.scene.to_dict()),
                       goal_predicates=goal, myopic_planner=True, max_repair_rounds=max_rounds,
                       council_stream=stream, events=events)
    plan = res.final_plan
    console.print(f"plan: VERIFIED={res.success}, {len(plan.steps)} steps, "
                  f"{len(rounds)-1} repairs, {llm.METRICS.total_model_s:.2f}s")
    metrics = {
        "model_s": round(llm.METRICS.total_model_s, 3),
        "calls": len(llm.METRICS.calls),
        "completion_tokens": sum(c.completion_tokens for c in llm.METRICS.calls),
        "tok_per_s": round(sum(c.completion_tokens for c in llm.METRICS.calls) /
                           max(1e-6, llm.METRICS.total_model_s), 1),
    }

    # ---- EXECUTE (capture frames + per-action ranges + predicate satisfaction) ----
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = TowerEnv(start_order=task.start_order, column_xy=(0.0, 0.0), robots="Panda",
                   controller_configs=cfg, has_renderer=False, has_offscreen_renderer=True,
                   use_camera_obs=False, control_freq=20, ignore_done=True, horizon=100000)
    env.boot(seed=seed)
    for _ in range(40):
        env.obs, _, _, _ = env.step(np.zeros(7))
    env.capture(camera)

    def preds_now():
        out = []
        h = env._cube_half
        for gp in goal:
            a, b = _color(gp.args[0]), _color(gp.args[1])
            pa, pb = env.cube_pos(a), env.cube_pos(b)
            ok = (abs(pa[0]-pb[0]) < 0.03 and abs(pa[1]-pb[1]) < 0.03
                  and abs(pa[2]-(pb[2]+2*h)) < 0.02)
            out.append(bool(ok))
        return out

    actions = []
    park_i = 0
    for s in plan.steps:
        if s.action != "place" or len(s.args) < 2:
            continue
        obj, target = _color(s.args[0]), s.args[1]
        f0 = len(env.frames)
        if target == "table":
            spot = _PARK_SPOTS[park_i % len(_PARK_SPOTS)]; park_i += 1
            env.pick_place(obj, "table", park_xy=spot, cam=camera)
            label = f"park {obj}"; kind = "park"
        else:
            env.pick_place(obj, _color(target), cam=camera)
            label = f"stack {obj} → {_color(target)}"; kind = "stack"
        actions.append({"label": label, "kind": kind, "obj": obj, "target": _color(target),
                        "f0": f0, "f1": len(env.frames), "preds": preds_now()})
        console.print(f"  {label}: frames {f0}-{len(env.frames)}  preds={actions[-1]['preds']}")

    native = all(preds_now())

    # downsample frames for embedding
    step = max(1, len(env.frames) // 120)
    keep_idx = list(range(0, len(env.frames), step))
    frames_b64 = [_jpeg_b64(env.frames[i]) for i in keep_idx]
    # remap action frame ranges into the downsampled index space
    idxmap = {orig: k for k, orig in enumerate(keep_idx)}

    def remap(f):
        ks = [k for orig, k in idxmap.items() if orig <= f]
        return ks[-1] if ks else 0
    for a in actions:
        a["k0"], a["k1"] = remap(a["f0"]), remap(a["f1"])

    trace = {
        "task": "Buried-Base Tower",
        "subtitle": f"long-horizon · n={n} · disassemble-then-rebuild",
        "n": n,
        "start_order": task.start_order,
        "goal_order": task.goal_order,
        "instruction": task.goal,
        "goal": [{"name": g.name, "args": g.args, "text": str(g)} for g in goal],
        "rounds": rounds,
        "success": bool(res.success),
        "metrics": metrics,
        "exec_frames": frames_b64,
        "actions": actions,
        "native": bool(native),
    }
    env.close()
    with open(OUT, "w") as f:
        json.dump(trace, f)
    console.print(f"\nwrote {OUT}: {len(frames_b64)} frames, {len(actions)} actions, "
                  f"native={native}, {metrics['tok_per_s']} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.n, seed=a.seed)


if __name__ == "__main__":
    main()
