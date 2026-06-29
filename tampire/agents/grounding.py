"""Grounding agent: detected objects -> symbolic predicates (PDDL-style)."""
from __future__ import annotations

import json
from typing import List

from .. import llm, prompts
from ..schemas import Predicate, Scene


def ground(scene: Scene) -> Scene:
    """Populate scene.predicates from scene.objects. Mutates and returns scene."""
    objects_json = json.dumps([o.to_dict() for o in scene.objects], indent=2)
    messages = [
        llm.sys(prompts.GROUNDING_SYS),
        llm.user(prompts.GROUNDING_USER.format(objects_json=objects_json)),
    ]
    data, _ = llm.chat_json(messages, label="grounding", temperature=0.1)
    preds: List[Predicate] = [Predicate.from_dict(p) for p in data.get("predicates", [])]
    scene.predicates = preds
    return scene
