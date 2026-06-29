"""Perception agent: image (pixels) -> WorldObjects, via Gemma 4 31B vision."""
from __future__ import annotations

from typing import List

from .. import llm, prompts
from ..schemas import Scene, WorldObject


def perceive(image_path: str, goal: str) -> Scene:
    """Run vision perception on an image and return a Scene (objects only;
    grounding fills predicates)."""
    messages = [
        llm.sys(prompts.PERCEPTION_SYS),
        llm.user_with_image(prompts.PERCEPTION_USER.format(goal=goal), image_path),
    ]
    data, _ = llm.chat_json(messages, label="perception", temperature=0.1)
    objects: List[WorldObject] = [WorldObject.from_dict(o) for o in data.get("objects", [])]
    return Scene(objects=objects, notes=data.get("notes", ""))
