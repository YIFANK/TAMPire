"""Planner agent: goal + symbolic state -> high-level Plan."""
from __future__ import annotations

from .. import llm, prompts
from ..schemas import Plan, Scene


def plan(scene: Scene, goal: str, *, myopic: bool = False) -> Plan:
    """Produce a high-level plan. ``myopic=True`` uses a greedy baseline planner
    that ignores preconditions — useful to demonstrate the repair council reliably."""
    predicates = "\n".join(f"  {p}" for p in scene.predicates) or "  (none)"
    object_ids = ", ".join(o.id for o in scene.objects)
    sys_prompt = prompts.PLANNER_MYOPIC_SYS if myopic else prompts.PLANNER_SYS
    messages = [
        llm.sys(sys_prompt),
        llm.user(prompts.PLANNER_USER.format(
            goal=goal, predicates=predicates, object_ids=object_ids)),
    ]
    data, _ = llm.chat_json(messages, label="planner", temperature=0.2)
    return Plan.from_dict(data)
