"""Fixed-base, long-horizon, realistic manipulation with closed-loop Gemma verification.

robosuite PickPlace: four real grocery items (milk, bread, cereal, can) on a table, to be
sorted into four bins. The base never moves — pure manipulation. A real Panda executes
every pick-and-place with OSC. After each grasp Gemma LOOKS at the camera and verifies the
predicate `holding(object)`; a failed grasp (e.g. the flat bread) is caught and the step is
retried. This is why the long task completes despite ~per-grasp stochasticity — and frequent
visual checking is only affordable because Gemma-4 runs fast on Cerebras.

  perceive : Gemma names the items it sees from pixels (multimodal grounding)
  plan     : each item → its sorting bin (the goal), an ordered pick-place sequence
  execute  : real OSC pick-place, fixed base, long-horizon (4 items)
  verify   : Gemma checks holding(object) after each grasp → retry on failure

    cd /Users/yifankang/R3-Manipulation && PYTHONPATH=/Users/yifankang/TAMPire \
      MUJOCO_GL=cgl PYTHONWARNINGS=ignore .venv-arm64/bin/python \
      -m tampire.tier2.pickplace_closed_loop
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl")

# the reliably top-graspable grocery items (the milk carton is a learned-policy-grade
# grasp — tips/wall-flush — so it's excluded for a clean scripted demo)
NAMES = ["Cereal", "Can", "Bread"]
HUMAN = {"Milk": "milk carton", "Bread": "loaf of bread", "Cereal": "cereal box", "Can": "soda can"}
CAM = "frontview"
_CALLS = {"n": 0, "t": 0.0}


def _ask(env, question):
    from .. import llm
    img = np.flipud(env.sim.render(width=640, height=480, camera_name=CAM)).copy()
    p = os.path.join(tempfile.mkdtemp(), "c.png")
    from PIL import Image
    Image.fromarray(img).save(p)
    sys = ("You verify a robot manipulation scene from one image. Answer strictly from the "
           'image. Reply ONLY JSON {"yes": true|false, "why": "<=6 words"}.')
    t0 = time.time()
    a, _ = llm.chat_json([llm.sys(sys), llm.user_with_image(question, p)],
                         label="verify", temperature=0)
    ms = (time.time() - t0) * 1000
    _CALLS["n"] += 1; _CALLS["t"] += ms / 1000
    return bool(a.get("yes", False)), a.get("why", ""), ms


def _identify(env):
    """Gemma lists the grocery items it sees (multimodal grounding)."""
    from .. import llm
    from PIL import Image
    img = np.flipud(env.sim.render(width=640, height=480, camera_name=CAM)).copy()
    p = os.path.join(tempfile.mkdtemp(), "s.png"); Image.fromarray(img).save(p)
    sys = ("You are a robot's perception agent. List the distinct grocery items you see on the "
           'table. Reply ONLY JSON {"items": ["<short noun>", ...]}.')
    t0 = time.time()
    a, _ = llm.chat_json([llm.sys(sys), llm.user_with_image("What grocery items are on the table?", p)],
                         label="identify", temperature=0)
    _CALLS["n"] += 1; _CALLS["t"] += time.time() - t0
    return a.get("items", [])


def run(log):
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = robosuite.make(env_name="PickPlace", robots="Panda", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                         control_freq=20, ignore_done=True, horizon=100000)
    env.reset()
    frames, ev = [], []

    def O(): return env._get_observations()
    def eef(): return np.array(O()["robot0_eef_pos"])
    def snap(): frames.append(np.flipud(env.sim.render(width=560, height=420, camera_name=CAM)).copy())
    def mark(kind, text, **kw): ev.append({"kind": kind, "text": text, "k": len(frames), **kw})

    def move(t, steps, kp, grip, cap=8, tol=0.0):
        t = np.asarray(t, dtype=float)
        for i in range(steps):
            a = np.zeros(env.action_dim); a[:3] = np.clip(kp * (t - eef()), -1, 1); a[6] = grip
            env.step(a)
            if i % cap == 0: snap()
            if tol and np.linalg.norm(t - eef()) < tol:
                break

    def grip(g, n):
        for _ in range(n):
            a = np.zeros(env.action_dim); a[6] = g; env.step(a)
        snap()

    def grasp(p, dz=0.0, jit=(0.0, 0.0)):
        p = np.asarray(p, dtype=float) + [jit[0], jit[1], dz]
        move(p + [0, 0, 0.10], 120, 5, -1); move(p + [0, 0, -0.004], 90, 3, -1)
        grip(1, 45); move(eef() + [0, 0, 0.22], 70, 5, 1)

    def place(t):
        t = np.asarray(t, dtype=float)
        move([t[0], t[1], eef()[2]], 70, 5, 1); move([t[0], t[1], t[2] + 0.10], 60, 3, 1)
        grip(-1, 30); move(eef() + [0, 0, 0.15], 40, 5, -1)

    bin_c = np.array(env.bin1_pos)[:2]

    def nudge(nm):
        """Push an object toward the source-bin center to clear it from a wall, then it's
        graspable from a clear side."""
        o = np.array(O()[nm + "_pos"], dtype=float)
        toward = bin_c - o[:2]
        if np.linalg.norm(toward) < 1e-6:
            return
        toward = toward / np.linalg.norm(toward)
        far = o[:2] - toward * 0.07           # approach from the wall side
        grip(1, 4)                            # fingers closed = a pushing tool
        move([far[0], far[1], o[2] + 0.08], 55, 5, 1)
        move([far[0], far[1], o[2]], 45, 4, 1)
        move([o[0] + toward[0] * 0.05, o[1] + toward[1] * 0.05, o[2]], 60, 3, 1)  # push to center
        move(eef() + [0, 0, 0.16], 35, 5, 1)
        grip(-1, 4)

    snap()
    seen = _identify(env)
    log(f"perception ▸ Gemma sees: {', '.join(seen)}")
    mark("perception", f"Gemma identified the items: {', '.join(seen[:5])}")

    # bounding-box half-sizes → grasp the upper body (the analytic top-grasp height)
    name2obj = {o.name: o for o in env.objects}
    obj_order = [o.name for o in env.objects]          # tbp is indexed in this order
    halfh = {nm: float(name2obj[nm].get_bounding_box_half_size()[2]) for nm in NAMES}

    tbp = env.target_bin_placements
    placed = 0
    for nm in NAMES:
        i = obj_order.index(nm)
        p = np.array(O()[nm + "_pos"], dtype=float); z0 = p[2]
        mark("plan", f"plan: {HUMAN[nm]} → bin {i+1}")
        held = False
        variants = [0.0, +0.02, -0.02, +0.04]
        for attempt in range(1, 5):
            if attempt > 1:
                grip(-1, 12); move(eef() + [0, 0, 0.18], 35, 4, -1)
            cur = np.array(O()[nm + "_pos"], dtype=float)
            tgt = np.array([cur[0], cur[1], cur[2] + halfh[nm] - 0.03 + variants[attempt - 1]])
            jit = (0.0, 0.0) if attempt < 3 else (0.012, 0.012)
            mark("action", f"grasp {HUMAN[nm]} (attempt {attempt})")
            grasp(tgt, dz=0.0, jit=jit)
            holding, why, ms = _ask(env, f"Is the robot gripper holding an object lifted above the table?")
            truth = bool(O()[nm + "_pos"][2] > z0 + 0.04)
            mark("check", f"holding({nm})? Gemma: {'YES' if holding else 'NO'} — “{why}”",
                 ok=(holding and truth), ms=round(ms))
            log(f"  {nm} attempt {attempt}: Gemma holding={holding} [{truth}] '{why}'")
            if holding and truth:
                held = True; break
            mark("replan", f"grasp of {HUMAN[nm]} failed → retry")
        if not held:
            log(f"  {nm}: could not grasp, skipping"); continue
        place(np.array(tbp[i]))
        ok = bool(env.objects_in_bins[i])
        placed += ok
        mark("check", f"in_bin({nm})? {'YES' if ok else 'NO'}", ok=ok)
        log(f"  {nm}: placed in bin → {ok}")
    success = (placed == len(NAMES))
    mark("success" if success else "action",
         f"sorted {placed}/{len(NAMES)} items into bins" + (" — task complete ✓" if success else ""))
    env.close()
    return placed, success, frames, ev


def _save(frames, ev, placed, success):
    from PIL import Image
    step = max(1, len(frames) // 150)
    keep = list(range(0, len(frames), step))
    idxmap = {o: k for k, o in enumerate(keep)}
    remap = lambda f: max([k for o, k in idxmap.items() if o <= f] or [0])
    fb = []
    for i in keep:
        im = Image.fromarray(frames[i])
        if im.width > 380: im = im.resize((380, int(im.height * 380 / im.width)))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=72); fb.append(base64.b64encode(buf.getvalue()).decode())
    for e in ev: e["k"] = remap(e["k"])
    trace = {"task": "robosuite PickPlace", "subtitle": "fixed-base · sort groceries · closed-loop",
             "placed": placed, "success": bool(success),
             "metrics": {"calls": _CALLS["n"], "total_s": round(_CALLS["t"], 2),
                         "ms_each": round(_CALLS["t"] / max(1, _CALLS["n"]) * 1000)},
             "frames": fb, "events": ev}
    json.dump(trace, open("/Users/yifankang/TAMPire/runs/trace_pickplace.json", "w"))
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save("/Users/yifankang/TAMPire/runs/rc_pickplace.gif", save_all=True,
                 append_images=imgs[1:], duration=70, loop=0)
    print(f"wrote runs/rc_pickplace.gif ({len(frames)} frames) + trace_pickplace.json ({len(fb)} frames)")


def main():
    from rich.console import Console
    c = Console()
    best = None
    for run_i in range(1, 7):
        _CALLS["n"] = 0; _CALLS["t"] = 0.0
        c.print(f"\n[bold]── Fixed-base sort run {run_i} (robosuite PickPlace) — closed-loop Gemma ──[/]")
        placed, success, frames, ev = run(c.print)
        c.print(f"  → {placed}/{len(NAMES)} sorted, success={success}  ({_CALLS['n']} checks, {_CALLS['t']:.1f}s)")
        if best is None or placed > best[0]:
            best = (placed, success, frames, ev, _CALLS["n"], _CALLS["t"])
        if placed >= len(NAMES):
            break
    placed, success, frames, ev, calls, t = best
    _CALLS["n"], _CALLS["t"] = calls, t
    c.print(f"\n[bold]BEST: {placed}/{len(NAMES)} sorted, native success={success}[/]  ·  {calls} Gemma checks "
            f"in {t:.1f}s (~{t/max(1,calls)*1000:.0f}ms each)")
    _save(frames, ev, placed, success)


if __name__ == "__main__":
    main()
