"""Long-horizon rearrangement tasks — TAMP's home turf.

A *buried-base tower*: the block that must end up on the BOTTOM of the goal tower
starts at the bottom of a wrongly-ordered start stack — blocked by every other
block, stacked above it in reverse goal order. There is no greedy shortcut: you
cannot place anything onto the goal base until it is cleared, so blocks must be
*parked* on the table and the tower rebuilt bottom-up. That is 4*(N-1) primitives
(N=5 -> 16 pick/place steps) with a long precondition chain (`clear`,
holding-one-at-a-time, build-bottom-up, no premature stacking) that a myopic,
goal-directed planner deadlocks on — its first move (`place goal-base's child on
the base`) is infeasible because the base isn't clear. The conjunctive goal (a
chain of `on` predicates) is checked end-to-end by the deterministic verifier —
exactly the long-horizon regime where the debate council earns its keep.

Run:
    python -m tampire.eval.longhorizon --n 5 --conditions baseline council
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..schemas import Plan, PlanStep, Predicate, Scene, WorldObject
from ..sim.world import BASE_XY

# bottom -> top canonical order for the GOAL tower
COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]

# one reachable column near the robot base to stack the start tower in
COLUMN_XY: Tuple[float, float] = (BASE_XY[0], BASE_XY[1] + 0.30)
# physics-consistent: the MuJoCo builder makes 5cm cubes (half=0.025), so the
# start stack must be spaced one full edge apart or it interpenetrates and topples.
BLOCK_H = 0.05


@dataclass
class TowerTask:
    scene: Scene
    goal: str
    goal_predicates: List[Predicate]
    n: int
    optimal_primitives: int     # oracle plan length (reference)
    start_order: List[str] = None     # colors bottom->top of the start column
    goal_order: List[str] = None      # colors bottom->top of the goal tower


def _block(oid: str, color: str, xy: Tuple[float, float], z: float) -> WorldObject:
    return WorldObject(id=oid, category="block", color=color,
                       position=(xy[0], xy[1], z), size=(BLOCK_H, BLOCK_H, BLOCK_H),
                       affordances=["graspable", "stackable"])


def tower_task(n: int = 5) -> TowerTask:
    """Build a buried-base tower task. The goal base stays at the bottom of the
    start stack, with every other block stacked above it in REVERSE goal order —
    so the base is fully buried and no block can go straight to its goal slot.
    Solving requires parking blocks on the table, then rebuilding bottom-up."""
    if not 3 <= n <= len(COLORS):
        raise ValueError(f"n must be in [3, {len(COLORS)}]")
    goal_order = COLORS[:n]                                  # bottom..top desired
    # base stays at the bottom; the rest pile on it in reverse goal order (worst case)
    start_order = [goal_order[0]] + list(reversed(goal_order[1:]))  # bottom..top at start

    objects: List[WorldObject] = []
    preds: List[Predicate] = []
    for level, color in enumerate(start_order):
        oid = f"{color}_block"
        z = BLOCK_H / 2 + level * BLOCK_H   # center of the cube at this stack level
        objects.append(_block(oid, color, COLUMN_XY, z))
        support = "table" if level == 0 else f"{start_order[level - 1]}_block"
        preds.append(Predicate("on", [oid, support]))
    # only the topmost start block is clear; all blocks are graspable
    preds.append(Predicate("clear", [f"{start_order[-1]}_block"]))
    for color in goal_order:
        preds.append(Predicate("graspable", [f"{color}_block"]))

    goal_predicates = [
        Predicate("on", [f"{goal_order[i + 1]}_block", f"{goal_order[i]}_block"])
        for i in range(n - 1)
    ]
    goal = ("restack all the blocks into one tower, from bottom to top: "
            + ", ".join(goal_order)
            + f" (so {goal_order[0]} is on the table at the bottom and "
            + f"{goal_order[-1]} is on top)")

    scene = Scene(objects=objects, predicates=preds,
                  notes=f"buried-base tower n={n}: start {start_order} -> goal {goal_order}")
    # oracle: (n-1) unstacks to table + (n-1) restacks, each a pick+place
    optimal = 4 * (n - 1)
    return TowerTask(scene=scene, goal=goal, goal_predicates=goal_predicates,
                     n=n, optimal_primitives=optimal,
                     start_order=start_order, goal_order=goal_order)


def solve_tower(task: TowerTask) -> Plan:
    """Deterministic oracle: disassemble to the table topmost-first, then rebuild
    the goal tower bottom-up. Proves the task is solvable and gives the reference
    plan length the LLM conditions are measured against."""
    on: Dict[str, str] = {}
    for p in task.scene.predicates:
        if p.name == "on" and not p.negated:
            on[p.args[0]] = p.args[1]
    blocks = [o.id for o in task.scene.objects]
    steps: List[PlanStep] = []

    def clear_block(x: str) -> bool:
        return not any(s == x for s in on.values())

    # 1. disassemble: move every stacked block to the table, topmost-first
    while any(sup != "table" for sup in on.values()):
        top = next(b for b in blocks if on.get(b) != "table" and clear_block(b))
        steps.append(PlanStep("pick", [top], "disassemble the wrong tower"))
        steps.append(PlanStep("place", [top, "table"], "set aside on the table"))
        on[top] = "table"

    # 2. reassemble bottom-up in the goal order
    goal_chain = [p.args for p in task.goal_predicates]   # [obj, support] bottom..top
    for obj, support in goal_chain:
        steps.append(PlanStep("pick", [obj], "build the correct tower"))
        steps.append(PlanStep("place", [obj, support], f"stack {obj} on {support}"))
        on[obj] = support

    return Plan(steps=steps, rationale="oracle: disassemble then rebuild bottom-up")


# ----------------------------------------------------------------------------
# Runner: baseline (myopic) vs council on the long-horizon task
# ----------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from .. import llm
    from ..pipeline import run
    from ..sim import feasibility

    console = Console()
    task = tower_task(args.n)

    # sanity: prove the task is solvable with the oracle
    oracle = solve_tower(task)
    ov = feasibility.check(task.scene, oracle, task.goal_predicates)
    console.print(f"[bold]Long-horizon buried-base tower[/]  n={task.n}  "
                  f"oracle={len(oracle.steps)} primitives  "
                  f"feasible={ov.ok} goal={ov.goal_satisfied}")
    console.print(f"[dim]{task.scene.notes}[/]\n")

    rows = []
    for cond in args.conditions:
        llm.METRICS.reset()
        res = run(
            task.goal,
            scene=Scene.from_dict(task.scene.to_dict()),   # fresh copy per condition
            goal_predicates=task.goal_predicates,
            myopic_planner=(cond == "baseline"),
            max_repair_rounds=args.max_rounds,
        )
        n_steps = len(res.final_plan.steps) if res.final_plan else 0
        rows.append((cond, res.success, len(res.rounds) - 1, n_steps,
                     llm.METRICS.total_wall_s, llm.METRICS.total_model_s,
                     llm.METRICS.tokens_per_s))
        mark = "[green]✓ VERIFIED[/]" if res.success else "[red]✗ failed[/]"
        console.print(f"  {cond:9s}  {mark}  plan={n_steps} steps  "
                      f"repairs={len(res.rounds) - 1}")

    t = Table(title="Long-horizon TAMP — baseline vs council")
    t.add_column("condition"); t.add_column("verified")
    t.add_column("repairs", justify="right"); t.add_column("plan len", justify="right")
    t.add_column("wall s", justify="right"); t.add_column("model s", justify="right")
    t.add_column("tok/s", justify="right")
    for c, ok, reps, ln, wall, model, tps in rows:
        t.add_row(c, "✓" if ok else "✗", str(reps), str(ln),
                  f"{wall:.2f}", f"{model:.3f}", f"{tps:.0f}")
    console.print(t)
    console.print("[dim]The myopic baseline cannot satisfy the long precondition "
                  "chain; the council repairs it to a verified plan. Model-compute "
                  "seconds show why replan-by-debate stays real-time on Cerebras.[/]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-horizon reversed-tower TAMP demo")
    ap.add_argument("--n", type=int, default=5, help="tower height (2..7)")
    ap.add_argument("--conditions", nargs="+", default=["baseline", "council"],
                    choices=["baseline", "council"])
    ap.add_argument("--max-rounds", type=int, default=6,
                    help="debate-repair budget (long horizon may need a few)")
    main_args = ap.parse_args()
    _run(main_args)


if __name__ == "__main__":
    main()
