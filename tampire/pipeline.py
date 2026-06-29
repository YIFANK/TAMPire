"""End-to-end TAMPire pipeline.

  image/scene + goal
    -> perception (if image)  -> grounding -> goal compilation
    -> plan -> feasibility check
    -> [debate council repair loop until feasible & goal-satisfied, or budget out]
    -> verified primitive sequence

Emits structured events through an optional callback so the demo can render each
stage live. Pure-logic; no printing here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agents import council, goalspec, grounding, perception, planner
from .config import CONFIG
from .schemas import FeasibilityResult, Plan, Predicate, Scene
from .sim import feasibility

# event callback: (stage, payload)
EventCB = Optional[Callable[[str, Dict[str, Any]], None]]


@dataclass
class RoundRecord:
    index: int
    plan: Plan
    verdict: FeasibilityResult


@dataclass
class PipelineResult:
    goal: str
    scene: Scene
    goal_predicates: List[Predicate]
    rounds: List[RoundRecord] = field(default_factory=list)
    final_plan: Optional[Plan] = None
    success: bool = False

    @property
    def primitives(self) -> List[Dict[str, Any]]:
        if not self.final_plan:
            return []
        return [{"action": s.action, "args": s.args} for s in self.final_plan.steps]


def load_scene(path: str) -> Scene:
    with open(path) as f:
        return Scene.from_dict(json.load(f))


def run(
    goal: str,
    *,
    scene: Optional[Scene] = None,
    image_path: Optional[str] = None,
    scene_path: Optional[str] = None,
    events: EventCB = None,
    council_stream: council.StreamCB = None,
    myopic_planner: bool = False,
    goal_predicates: Optional[List[Predicate]] = None,
    max_repair_rounds: Optional[int] = None,
) -> PipelineResult:
    def emit(stage: str, **payload: Any) -> None:
        if events:
            events(stage, payload)

    # ---- 1. acquire scene -------------------------------------------------
    if scene is None and scene_path is not None:
        scene = load_scene(scene_path)
    if scene is not None and not scene.predicates and not image_path:
        # a JSON scene with objects but no predicates -> ground it
        emit("grounding_start")
        grounding.ground(scene)
        emit("grounding_done", scene=scene)
    if scene is None:
        if not image_path:
            raise ValueError("Provide one of: scene, scene_path, image_path")
        emit("perception_start", image=image_path)
        scene = perception.perceive(image_path, goal)
        emit("perception_done", scene=scene)
        emit("grounding_start")
        grounding.ground(scene)
        emit("grounding_done", scene=scene)

    # ---- 2. compile goal --------------------------------------------------
    # eval can inject ground-truth predicates to remove the goalspec LLM confound
    if goal_predicates is None:
        emit("goalspec_start")
        goal_predicates = goalspec.compile_goal(scene, goal)
    emit("goalspec_done", goal_predicates=goal_predicates)

    result = PipelineResult(goal=goal, scene=scene, goal_predicates=goal_predicates)

    # ---- 3. initial plan --------------------------------------------------
    emit("plan_start", round=0)
    plan = planner.plan(scene, goal, myopic=myopic_planner)
    verdict = feasibility.check(scene, plan, goal_predicates)
    result.rounds.append(RoundRecord(0, plan, verdict))
    emit("plan_done", round=0, plan=plan, verdict=verdict)

    if verdict.ok and verdict.goal_satisfied:
        result.final_plan = plan
        result.success = True
        emit("success", round=0, plan=plan)
        return result

    # ---- 4. debate-repair loop -------------------------------------------
    rounds_budget = CONFIG.max_repair_rounds if max_repair_rounds is None else max_repair_rounds
    for r in range(1, rounds_budget + 1):
        emit("debate_start", round=r, verdict=verdict)
        plan = council.debate_and_repair(
            scene, goal, plan, verdict, stream_cb=council_stream
        )
        verdict = feasibility.check(scene, plan, goal_predicates)
        result.rounds.append(RoundRecord(r, plan, verdict))
        emit("repair_done", round=r, plan=plan, verdict=verdict)
        if verdict.ok and verdict.goal_satisfied:
            result.final_plan = plan
            result.success = True
            emit("success", round=r, plan=plan)
            return result

    # ---- 5. budget exhausted ---------------------------------------------
    result.final_plan = plan
    result.success = False
    emit("failure", rounds=rounds_budget, plan=plan, verdict=verdict)
    return result
