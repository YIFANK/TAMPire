# TAMPire 🔥

**A Multi-Agent Zero-Shot Robotics Planner from Pixels**

TAMPire uses **Gemma 4 31B on Cerebras** as a fast multi-agent council that converts
an image + a natural-language goal into a *verified* task-and-motion plan for tabletop
robots — then repairs that plan through agent debate when motion checks fail.

The key bet: Cerebras inference is fast enough (~7ms/call of model compute) that
**iterative replan-by-debate becomes real-time** — an architecture that is normally
too slow to be practical.

```
  image + goal
       │
       ▼
 ┌───────────────┐   pixels → objects, relations, affordances
 │  Perception   │   (Gemma 4 31B, vision)
 └──────┬────────┘
        ▼
 ┌───────────────┐   scene → symbolic predicates (PDDL-ish)
 │  Grounding    │
 └──────┬────────┘
        ▼
 ┌───────────────┐   goal + predicates → high-level plan
 │   Planner     │◀──────────────┐
 └──────┬────────┘                │ repair
        ▼                         │
 ┌───────────────┐   geometric feasibility (collision / reach / preconds)
 │  Sim Checker  │   lightweight, no physics engine needed
 └──────┬────────┘                │
        ▼  infeasible?            │
 ┌───────────────┐   council debates the failure and proposes a fix
 │  Debate       │────────────────┘
 │  Council      │
 └──────┬────────┘
        ▼  feasible
   primitive sequence  →  pick(red_block) place(red_block, bowl) ...
```

## Why this is more than "LLM + robotics"
- **Multimodal grounding** — perception runs on the actual image, not a hand-written scene.
- **Visible agent collaboration** — the debate council's messages stream live in the demo.
- **Closed verification loop** — a geometric simulator gates every plan; the LLM never
  gets the last word unverified.
- **Speed-dependent** — the whole point is iterating fast. There's a latency counter on screen.

## Status
This is a hackathon scaffold. The end-to-end loop runs today on a JSON scene and on a
real image. The simulator is an intentionally lightweight geometric stub (not PyBullet) —
it returns the *reason* a step is infeasible, which is exactly the signal the debate
council repairs against.

## Quickstart
```bash
# Use Python 3.11/3.12 — the Tier-1 MuJoCo sim has no cp39 wheel on recent macOS.
python3.12 -m venv .venv312 && source .venv312/bin/activate
pip install -r requirements.txt -r requirements-sim.txt
# .env must contain CEREBRAS_API_KEY=...

python scripts/smoke_test.py            # verify API + vision work

# end-to-end from a JSON scene
python -m tampire.demo --scene scenes/blocks.json --goal "put the red block in the bowl"

# end-to-end from PIXELS (runs vision perception)
python -m tampire.demo --image scenes/scene.png  --goal "put the blue block in the bowl"

# THE DEMO: --baseline uses a myopic planner that ignores preconditions, so the
# verifier catches a bad plan and you watch the 3-critic council repair it live.
python -m tampire.demo --scene scenes/blocks.json --goal "put the red block in the bowl" --baseline

# --render writes a top-down animation of the verified plan executing
python -m tampire.demo --scene scenes/blocks.json --goal "put the red block in the bowl" \
    --baseline --render runs/redbowl     # -> runs/redbowl.gif + per-step PNGs
```

`blocks.json` is rigged so `green_block` sits on `red_block`. A naive plan tries to
`pick(red_block)` while it isn't clear; the simulator rejects it with a precise reason,
and the council inserts the clearing steps. That's the "watch it fail, watch it fix"
moment for judges.

> Note: on macOS system Python you may see a harmless `urllib3 NotOpenSSLWarning`
> (LibreSSL). Suppress with `PYTHONWARNINGS=ignore`.

## Evaluation

**Tier 0 — symbolic ablation (no extra deps).** Procedurally generates solvable
scenes and reports success rate / repairs / latency per condition. Success is judged
by the deterministic verifier, *not* the LLM, and an oracle confirms every task is
solvable. The headline: the debate council recovers the myopic baseline's failures.
```bash
python -m tampire.eval --n 12 --json runs/eval.json
#  oracle  100%   baseline 75%   council 100%   smart 100%
#  Debate council lifts success 75% -> 100% by repairing infeasible plans.
```

**Tier 1 — MuJoCo, from real pixels (needs `requirements-sim.txt`).** Builds a real
physics tabletop, renders it, plans *from the rendered image*, executes the plan back
in physics, and checks success from the simulated body positions — an independent
verifier. Reports whether the symbolic and physics verifiers agree.
```bash
python -m tampire.simreal --seed 1010 --baseline --out runs/mjrun   # single run + GIF
python -m tampire.simreal --scene scenes/blocks.json --goal "put the red block in the bowl"
#  symbolic verifier : PASS / physics verifier : PASS / verifiers agree: yes

# quantitative physics benchmark over N scenes (the full pixel->plan->physics loop;
# also pays the perception-error tax that Tier 0 never saw):
python -m tampire.simreal.bench --n 10 --conditions baseline council --json runs/bench_phys.json
```

**Tier 2 — robosuite (recognized benchmark, real Panda arm; needs `requirements-tier2.txt`).**
TAMPire perceives robosuite's real `agentview` pixels, plans, and drives the Panda through
a scripted OSC pick-place skill on the **Stack** task; success is robosuite's *native*
criterion. (PyBullet benchmarks — VIMA-Bench, Ravens — can't build on this macOS/Python,
so the MuJoCo-based robosuite is the chosen one. `mujoco` is pinned to `3.3.0`.)
```bash
pip install -r requirements-tier2.txt
python -m tampire.tier2 --seed 0           # privileged cube poses
python -m tampire.tier2 --seed 0 --vision  # grasp from MULTI-VIEW VISION poses (no privileged state)
#  TAMPire symbolic verifier : PASS
#  robosuite native success  : PASS
#  pose source               : VISION (multi-view triangulation)  -> runs/rs.gif
```
`--vision` triangulates the cube poses from robosuite's agentview/frontview/sideview via the
same multi-agent council, then drives the real Panda to the *estimated* poses — closing the
loop with no privileged state.

**Tier 3 — NVIDIA RoboLab (`tampire/tier3/`).** RoboLab is an Isaac-Lab benchmark of 120
manipulation tasks (block stacking, bowl-in-bin, mug arrangement) — TAMPire's domain.
`TAMPireClient` implements RoboLab's `InferenceClient` ABC: per episode it perceives the
camera frame + instruction, runs the TAMPire pipeline to a verified plan, and drives a
skill controller emitting RoboLab's per-step action chunks.
> RoboLab needs an **NVIDIA RTX GPU + Linux + Isaac Sim**, so it can't run on this Mac.
> The adapter is validated against RoboLab's *real* interface (ABC read from source,
> vendored for off-hardware import) and **self-tested without Isaac**:
```bash
python -m tampire.tier3.selftest    # ABC conformance + real planning + skill convergence
# -> PASS — verified off-hardware
# On RTX hardware: python -m tampire.tier3.run --task StackYellowOnRedTask --num-envs 10
```
See [tampire/tier3/README.md](tampire/tier3/README.md) for the hardware notes and the two
task-specific hooks (`action_spec`, `pose_provider`).

## Vision-based grounding — multi-agent 3D from pixels (`tampire/perception3d/`)
The piece that makes TAMPire run with **no privileged state**: a council of Gemma-4
vision agents localizes each object's base across **multiple camera views**, and the
pixels are **triangulated** into metric 3D. Object stacking then falls out of the
estimated height rather than a noisy per-image label. Those poses feed TAMP — the
symbolic plan, the geometric reach/collision feasibility, and the motion waypoints are
all computed from *perceived* geometry.

```bash
python -m tampire.perception3d.bench --n 10 --multiview --agents 2
```
| variant (n=10) | pose error | stack-detection | plan→physics |
|---|---|---|---|
| single angled view | 4.0 cm | 50% | 67% |
| 3-view triangulation | 1.6 cm | 70% | 70% |
| **depth + semantic council** (TiPToP-style) | 3.3 cm | **90%** | **100%** |

The full pipeline `render → council + depth → object-centric scene → TAMP → execute →
physics-verified` runs end-to-end on perceived state, no privileged poses.

### TiPToP-style two-branch perception (`perception3d/depth.py`, `collision.py`)
Inspired by [TiPToP](https://github.com/tiptop-robot/tiptop), but every component picked to
run **GPU-free** on this machine (TiPToP's FoundationStereo / M2T2 / cuTAMP need an NVIDIA
GPU; Gemini-ER would replace our Gemma thesis — all skipped):
- **Semantic branch** — the Gemma-4 council says *which* objects exist (block vs bowl) and
  grounds the goal.
- **3D branch** — the **simulator's depth render** (replacing FoundationStereo) + **colour
  masks** (replacing SAM) give per-object point clouds; centroids are poses, **z-extents
  give stacking geometrically** (→ 90% vs the VLM-label 70%), and **`scipy.ConvexHull`
  builds collision meshes** (TiPToP's "over-approximate hulls are surprisingly sufficient").

Those meshes feed real **collision checking** — exact convex-hull intersection via an LP
(no `fcl`/`trimesh`). On perceived meshes it correctly flags an occupied target and clears
an empty bowl:
```
place blue_block ON red_block  (occupied by purple): COLLIDES with ['purple_block']
place blue_block IN green_bowl (empty)            : free
```
Reading stacking off depth instead of VLM labels is what pushed plan→physics to 100% on the
n=10 set. (Triangulation, the no-depth fallback, still cuts monocular error 2.5× and is the
path for cameras without a depth channel.)

### Finding: symbolic verification is necessary but not sufficient
A council can drive *symbolic* success to 100%, but physics success is capped by
**perception**: when a stacked/occluded block is missed, the plan is valid in the
perceived world yet fails in physics — only the independent physics verifier catches it.
That's why the work above matters: moving from a single view to **multi-view
triangulation** cuts pose error ~2.5× and lifts stack detection 50% → 70% by fixing the
*grounding*, not the planner. The remaining gap is honest residual perception error
(stacked-block detection).

## Layout
```
tampire/
  config.py        env + model config
  llm.py           Cerebras client (text + vision), JSON mode, timing
  schemas.py       Scene / Object / Predicate / PlanStep / Feasibility dataclasses
  prompts.py       all agent prompts in one place
  agents/
    perception.py  image → objects, relations, affordances
    grounding.py   scene → symbolic predicates
    planner.py     goal + predicates → high-level plan
    council.py     debate + repair on feasibility failure
    goalspec.py    NL goal → target predicates (checked deterministically)
  sim/
    world.py       geometric world state from objects
    feasibility.py geometric checks → pass/fail + reason (the verifier)
  pipeline.py      end-to-end orchestration
  demo.py          CLI with live streaming + latency counter
  render.py        top-down matplotlib render + GIF of a plan executing
  eval/            Tier-0: scenegen, oracle, ablation harness   (python -m tampire.eval)
  simreal/         Tier-1: MuJoCo env, render-from-pixels runner, physics bench
                          (python -m tampire.simreal ; python -m tampire.simreal.bench)
  tier2/           Tier-2: robosuite Stack + scripted Panda skill (python -m tampire.tier2)
  tier3/           Tier-3: RoboLab InferenceClient adapter + off-hardware self-test
                          (python -m tampire.tier3.selftest ; tier3/run.py on RTX hardware)
  perception3d/    no-privileged-state grounding. camera + triangulation (estimator.py),
                          TiPToP-style depth branch (depth.py: point clouds, hulls, stacking),
                          convex-hull collision (collision.py), two-branch fuse (perceive.py),
                          bench (python -m tampire.perception3d.bench --depth)
scenes/            sample JSON scenes + a generated image
scripts/           smoke_test.py, make_scene_image.py
```

## References (design inspiration, not dependencies)
- TAMP system: tiptop — https://github.com/tiptop-robot/tiptop
- Multi-agent robotics autoresearch: NVIDIA ENPIRE — https://research.nvidia.com/labs/gear/enpire/
- Simulation benchmark to graduate to: NVIDIA RoboLab — https://research.nvidia.com/labs/srl/projects/robolab/
