# TAMPire — Demo Assets & 60-Second Script

**Track 1: Multiverse Agents** — Best Multi-Agent + Multimodal Use Case.
Thesis: *Gemma-4's Cerebras inference speed makes closed-loop, perception-in-the-loop
robot execution practical — a real arm re-checks its symbolic state from pixels after
every action and replans when it fails.*

## The 60-second story (recommended order)

1. **Hook — the recovery (≈25s).** Play `runs/recover_dashboard.html`.
   - A real Panda picks a food object (RoboCasa). Grasp attempt 1 misses.
   - **Gemma-4 looks at the camera → `holding(object)? NO — "food object is on counter"`** → the
     system **REPLANS** and retries → `holding YES` → places in sink → `in(object,sink) YES`
     → **RoboCasa native success**. Predicate checkboxes flip live; ms/check counter shows ~0.7s.
2. **Why it's possible — the speed (≈20s).** Play `runs/speed_race.html`.
   - Same model (gemma-4-31b) both sides: **Cerebras 252 vs GPU 90 tok/s = 2.8× throughput**.
   - Punchline: *only at this latency can you afford to re-perceive after every action.*
3. **Range — long-horizon execution (≈15s).** Show `runs/tower_arm_n5.gif` and
   `runs/rc_vision_mobile.gif` as B-roll: a 16-step real-arm task and a zero-shot
   vision→drive→grasp→place, both RoboCasa-scored successes.

## Assets

| file | what it shows | status |
|---|---|---|
| `runs/recover_dashboard.html` | **centerpiece** — closed-loop: Gemma catches failed grasp from pixels → replan → retry → success; live predicates + speed | ✅ |
| `runs/rc_recover.gif` / `_small.gif` | the recovery execution itself (RoboCasa native PASS) | ✅ |
| `runs/speed_race.html` | animated Cerebras-vs-GPU side-by-side on the real debate workload | ✅ |
| `runs/speed_compare.json` | raw latency/throughput numbers | ✅ |
| `runs/rc_pickplace.gif` / `_small.gif` | **fixed-base** long-horizon sort — Panda sorts groceries (cereal/can/bread) into bins, real OSC, base never moves, each grasp Gemma-verified; clean 3/3 | ✅ |
| `runs/rc_vision_mobile.gif` | Gemma multi-cam localization (0.5cm) + real velocity driving + grasp + place; native PASS | ✅ |
| `runs/tower_arm_n5.gif` | long-horizon (16-step) real Panda OSC execution; native PASS | ✅ |
| `runs/tampire_dashboard.html` | PackFoodByTemp multi-agent *planning* view (optional/secondary) | ✅ |

## How to run / regenerate

```bash
# closed-loop recovery (sweeps seeds, writes rc_recover.gif + trace_recover.json)
cd /Users/yifankang/R3-Manipulation && PYTHONPATH=/Users/yifankang/TAMPire \
  MUJOCO_GL=cgl PYTHONWARNINGS=ignore .venv-arm64/bin/python \
  -m tampire.robocasa.closed_loop_pick

# speed comparison (writes speed_compare.json)
cd /Users/yifankang/TAMPire && .venv-arm64/bin/python -m tampire.eval.speed_compare

# dashboards are static HTML in runs/ — open in a browser (or serve runs/ on :8765)
```

## Judging-criteria mapping
- **Agent collaboration:** planner → 3 critics → repair chair (debate); at execution time a
  perception agent verifies each action and triggers replanning.
- **Multimodal:** Gemma-4-31B reads the scene from RoboCasa camera images (localization +
  predicate verification from pixels).
- **Speed in Action:** `speed_race.html` / demo speed clip — 2.8× higher throughput on the SAME model; this is what makes
  per-action perception checks affordable.
- **Innovation:** physical-AI / robotics; closed-loop TAMP with a VLM in the verification loop.

## Honesty notes (for the script / Q&A)
- In the recovery clip the **first grasp's failure is induced** (a deliberate depth error) so the
  recovery arc shows on every take. The **detection, replan, retry, and success are all real**, and
  grasping is per-layout stochastic (we sweep seeds for a clean end-to-end run).
- **PackFoodByTemp** is shown as a *verified plan*, not a physical execution (4m kitchen driving +
  fridge/pan grasps aren't reliable in sim). The *executed* tasks are the recovery pick, the mobile
  pick-and-place, and the tower.
- **Speed comparison:** the SAME model (`google/gemma-4-31b-it`) runs on both Cerebras and
  Together's GPU — apples-to-apples. We report **throughput (252 vs 90 tok/s = 2.8×)** rather than
  raw wall-clock, because Together's serving didn't emit a natural stop (generated to the token cap
  every call), which would inflate a wall-clock number unfairly.
- **Fixed-base sort:** the robosuite milk carton is excluded — it's a learned-policy-grade grasp
  (tall, tips, spawns wall-flush); scripted top-grasp reliably handles cereal/can/bread (clean 3/3).
  Run `python -m tampire.tier2.pickplace_closed_loop` (sweeps runs, banks the best).
