"""Compile a natural-language goal into target predicates (TAMP-style goal spec).

Done once, up front. The simulator then checks goal satisfaction deterministically
against these predicates — the LLM never gets to *declare* success unverified.
"""
from __future__ import annotations

from typing import List

from .. import llm, prompts
from ..schemas import Predicate, Scene


def compile_goal(scene: Scene, goal: str) -> List[Predicate]:
    object_ids = ", ".join(o.id for o in scene.objects)
    messages = [
        llm.sys(prompts.GOAL_SYS),
        llm.user(prompts.GOAL_USER.format(goal=goal, object_ids=object_ids)),
    ]
    data, _ = llm.chat_json(messages, label="goalspec", temperature=0.0)
    return [Predicate.from_dict(p) for p in data.get("goal_predicates", [])]
