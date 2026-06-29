# TAMPire — Submission

**A Multi-Agent Zero-Shot Robotics Planner from Pixels** · Gemma-4-31B × Cerebras
*Track 1: Multiverse Agents (Best Multi-Agent + Multimodal Use Case)*

---

## 60-second narration script (timed to `runs/TAMPire_demo_60s.mp4`)

**[0:00–0:03 · Title]**
"TAMPire turns a vision-language model into a robot that plans, acts, and recovers — from pixels."

**[0:03–0:21 · Closed-loop recovery]**
"A real robot arm picks a food item. The first grasp misses — and Gemma-4, looking only at the
camera, catches it: *object still on the counter*. The system replans the trajectory, retries,
and succeeds — dropping it in the sink for a RoboCasa-native success. Every check is one fast
Cerebras call."

**[0:21–0:35 · Fixed-base long-horizon sort]**
"Same idea, longer horizon: the arm sorts groceries into bins, base fixed, and Gemma verifies
every grasp from pixels — so the multi-step task completes even when individual grasps are hard."

**[0:35–0:47 · Speed]**
"This only works because Gemma-4 runs on Cerebras. The exact same model — gemma-4-31b — generates
plans nearly three times faster than on a GPU provider, 252 versus 90 tokens per second — fast
enough to re-check perception after *every* action."

**[0:47–0:54 · More execution]**
"Zero-shot mobile manipulation — perceive, drive, grasp, place. And a sixteen-step long-horizon
tower, executed by a real arm. All scored by RoboCasa's own success checks."

**[0:54–end · Close]**
"TAMPire: multi-agent planning, multimodal perception, Cerebras speed, real physical AI."

---

## Devpost write-up

### Inspiration
LLM robot planners usually plan once and hope. Real robots fail mid-task — a grasp slips, an
object isn't where it was. We wanted to use a fast multimodal model not just to *plan*, but to
*watch* execution and recover — which is only practical if inference is fast enough to call
after every action.

### What it does
TAMPire drives a robot in RoboCasa / robosuite end-to-end:
- **Perceive** — Gemma-4 reads the scene from camera pixels (object localization, hot/cold and
  item classification, predicate checks).
- **Plan** — a multi-agent council (planner → 3 critics with distinct lenses → repair chair)
  produces a verified task-and-motion plan; for mobile tasks it inserts navigation steps.
- **Execute** — a real Panda arm runs the plan with OSC (real contact physics).
- **Recover** — after each action Gemma verifies the relevant predicate from pixels; on failure
  the step is replanned and retried. Scored by RoboCasa's native success check.

### How we built it
Cerebras OpenAI-compatible API (`gemma-4-31b`, multimodal) for all perception/planning/verification;
RoboCasa + robosuite + MuJoCo for the worlds and real OSC control; a symbolic world model +
feasibility checker as the planning substrate the council debates over.

### Challenges
- Mobile-base control in robosuite is a unicycle that must be re-calibrated after every turn.
- Grasping arbitrary objects with scripted OSC is stochastic — which is exactly why the
  closed-loop visual verification matters (catch + retry).
- Verifying *success* of a small in-gripper object is hard; verifying *location* (on counter / in
  bin / in sink) from pixels is reliable, so we check those.

### Accomplishments
Closed-loop failure→recovery→success on a real food object (native PASS); fixed-base long-horizon
grocery sort (3/3, every grasp Gemma-verified); zero-shot vision+drive+grasp+place; a 16-step
long-horizon tower; and a measured **2.8× throughput gain** (252 vs 90 tok/s) running the *same*
gemma-4-31b model on Cerebras vs a GPU provider.

### What's next
Learned grasp policies for the hard objects (milk carton), full mobile long-horizon execution,
and richer multi-step kitchen tasks.

---

## Honesty / disclosure notes (for Q&A)
- In the recovery clip the **first grasp's failure is induced** (a deliberate depth error) for a
  repeatable take; detection, replanning, retry, and success are all real. Grasping is per-layout
  stochastic, so we sweep seeds for a clean end-to-end run.
- **PackFoodByTemp** is shown as a verified *plan*, not physical execution (4 m kitchen driving +
  fridge/pan grasps aren't reliable in sim).
- The **milk carton** is excluded from the fixed-base sort — it's a learned-policy-grade grasp.
- **Speed baseline:** the SAME model (`google/gemma-4-31b-it`) runs on both Cerebras and Together's
  GPU serving, so it's apples-to-apples. We report **throughput (252 vs 90 tok/s = 2.8×)** rather
  than raw wall-clock, because Together's serving didn't emit a natural stop (it generated to the
  token cap every call), which would inflate a wall-clock number unfairly.

## Files
- `runs/TAMPire_demo_60s.mp4` — the assembled video (rebuild: `python -m tampire.tier2.make_demo_video`)
- `runs/recover_dashboard_vid.mp4`, `runs/pickplace_dashboard.mp4` — per-beat dashboard clips
- `runs/speed_compare.json` — raw latency/throughput numbers
- `DEMO.md` — asset index + storyboard
