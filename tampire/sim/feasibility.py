"""Feasibility checker: symbolically execute a Plan in a fresh World and report
per-step results plus whether the goal holds at the end."""
from __future__ import annotations

from typing import List

from ..schemas import FeasibilityResult, Plan, Predicate, Scene, StepCheck
from .world import World


def check(scene: Scene, plan: Plan, goal_predicates: List[Predicate]) -> FeasibilityResult:
    world = World.from_scene(scene)
    checks: List[StepCheck] = []
    all_ok = True

    for i, step in enumerate(plan.steps):
        ok, reason = world.apply(step)
        checks.append(StepCheck(index=i, ok=ok, reason=reason))
        if not ok:
            all_ok = False
            break  # stop at first failure; that's the one the council repairs

    goal_satisfied = all_ok and world.satisfies(goal_predicates)
    return FeasibilityResult(ok=all_ok, checks=checks, goal_satisfied=goal_satisfied)
