"""Deterministic oracle solver — knows the task structure and emits a correct
plan. Used to (a) prove every generated task is solvable and (b) provide a
100%-reference upper bound the LLM conditions are measured against."""
from __future__ import annotations

from typing import List, Optional

from ..schemas import Plan, PlanStep, Predicate, Scene
from .scenegen import EvalTask


def _blocker_of(scene: Scene, target: str) -> Optional[str]:
    for p in scene.predicates:
        if not p.negated and p.name == "on" and len(p.args) == 2 and p.args[1] == target:
            return p.args[0]
    return None


def solve(task: EvalTask) -> Plan:
    scene, gp = task.scene, task.goal_predicates[0]
    steps: List[PlanStep] = []

    def clear(obj: str) -> None:
        b = _blocker_of(scene, obj)
        if b:
            steps.append(PlanStep("pick", [b], "clear the target"))
            steps.append(PlanStep("place", [b, "table"], "set blocker aside"))

    if gp.name == "in":
        obj, bowl = gp.args
        clear(obj)
        steps.append(PlanStep("pick", [obj]))
        steps.append(PlanStep("place", [obj, bowl]))
    elif gp.name == "on":
        a, b = gp.args
        clear(a)
        clear(b)
        steps.append(PlanStep("pick", [a]))
        steps.append(PlanStep("place", [a, b]))
    return Plan(steps=steps, rationale="oracle")
