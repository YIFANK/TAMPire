"""Record a full TAMPire run into a JSON trace the dashboard plays back.

Captures, for an official RoboCasa long-horizon task (PackFoodByTemp):
  • the camera observation (base64 PNG),
  • Gemma's multimodal hot/cold classification,
  • the goal predicates,
  • every debate round: the plan, the simulator verdict (which step failed & why),
    each critic's diagnosis, the repair-chair's fix, and per-predicate satisfaction,
  • Cerebras timing/throughput.

    cd /Users/yifankang/R3-Manipulation && PYTHONPATH=/Users/yifankang/TAMPire \
      MUJOCO_GL=cgl PYTHONWARNINGS=ignore .venv-arm64/bin/python \
      -m tampire.robocasa.demo_trace --seed 3
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl")

OUT = "/Users/yifankang/TAMPire/runs/trace_pfbt.json"


def _pred_status(scene, plan, goal_preds):
    from ..sim.world import World
    w = World.from_scene(scene)
    for s in plan.steps:
        ok, _ = w.apply(s)
        if not ok:
            break
    return [bool(w.satisfies([p])) for p in goal_preds]


def run(seed):
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env
    from PIL import Image
    from .. import llm, pipeline
    from .longhorizon_official import _classify, _build_scene

    env = create_env(env_name="PackFoodByTemp", seed=seed, render_onscreen=False)
    env.reset()
    instr = env.get_ep_meta()["lang"].strip()

    # observation image
    img = np.flipud(env.sim.render(width=720, height=540, camera_name="robot0_agentview_center")).copy()
    p = os.path.join(tempfile.mkdtemp(), "obs.png")
    Image.fromarray(img).save(p)
    with open(p, "rb") as f:
        obs_b64 = base64.b64encode(f.read()).decode()

    classes = _classify(env, instr, ["hot0", "hot1", "cold0", "cold1"])
    scene, goal, base = _build_scene(env, classes)

    obs = env._get_observations()
    bxy = np.array(obs["robot0_base_pos"])[:2]
    dists = {k: float(np.linalg.norm(np.array(obs[k + "_pos"])[:2] - bxy))
             for k in ("hot0", "hot1", "cold0", "cold1", "tupperware0", "tupperware1")}

    # ---- capture callbacks ----
    rounds = []                 # one entry per debate round
    cur = {"critics": []}       # being filled
    state = {"buf": [], "persona": None}

    def stream(ev, who, delta):
        if ev == "critic_start":
            state["persona"] = who.split("—")[0].strip(); state["buf"] = []
        elif ev == "critic_delta":
            state["buf"].append(delta)
        elif ev == "critic_end":
            cur["critics"].append({"persona": state["persona"], "text": "".join(state["buf"]).strip()})
        elif ev == "repair_start":
            cur["repairing"] = True

    def events(stage, payload):
        if stage == "plan_done":          # round 0 (initial myopic plan)
            rounds.append(_mk_round(0, payload["plan"], payload["verdict"], [], scene, goal))
        elif stage == "debate_start":
            cur["critics"] = []
        elif stage == "repair_done":
            rounds.append(_mk_round(payload["round"], payload["plan"], payload["verdict"],
                                    list(cur["critics"]), scene, goal))

    llm.METRICS.reset()
    clean = ("Pack each food item into its assigned tupperware: the hot items into "
             "tupperware0 and the cold items into tupperware1. The tupperwares are "
             "already in place and must not be moved.")
    res = pipeline.run(clean, scene=scene, goal_predicates=goal, myopic_planner=True,
                       max_repair_rounds=10, council_stream=stream, events=events)

    trace = {
        "task": "PackFoodByTemp",
        "subtitle": "official RoboCasa long-horizon · sort food by temperature",
        "instruction": instr,
        "obs_image": obs_b64,
        "classification": [
            {"id": fid, "food": classes.get(fid, ("?", "?"))[1],
             "temp": classes.get(fid, ("?", "?"))[0],
             "target": "tupperware0" if classes.get(fid, ("hot",))[0] == "hot" else "tupperware1"}
            for fid in ("hot0", "hot1", "cold0", "cold1")
        ],
        "distances": dists,
        "goal": [{"name": g.name, "args": g.args, "text": str(g)} for g in goal],
        "rounds": rounds,
        "success": bool(res.success),
        "metrics": {
            "model_s": round(llm.METRICS.total_model_s, 3),
            "calls": len(llm.METRICS.calls),
            "completion_tokens": sum(c.completion_tokens for c in llm.METRICS.calls),
            "tok_per_s": round(sum(c.completion_tokens for c in llm.METRICS.calls) /
                               max(1e-6, llm.METRICS.total_model_s), 1),
        },
    }
    env.close()
    with open(OUT, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"wrote {OUT}: {len(rounds)} rounds, success={res.success}, "
          f"{trace['metrics']['tok_per_s']} tok/s")


def _mk_round(idx, plan, verdict, critics, scene, goal):
    failed = next((c for c in verdict.checks if not c.ok), None)
    return {
        "round": idx,
        "steps": [{"action": s.action, "args": s.args,
                   "move_base": s.action == "move_base"} for s in plan.steps],
        "verdict_ok": bool(verdict.ok and verdict.goal_satisfied),
        "failed_step": (failed.index + 1) if failed else None,
        "failed_reason": failed.reason if failed else None,
        "critics": critics,
        "pred_satisfied": _pred_status(scene, plan, goal),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    run(ap.parse_args().seed)


if __name__ == "__main__":
    main()
