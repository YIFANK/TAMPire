"""Kitchen perception for the RoboCasa adapter.

A Gemma-4 vision agent looks at the real rendered frame and, given the task
instruction as context, identifies the graspable object, the surface it rests on,
and the target receptacle (and whether that receptacle is a closed openable
container). We turn that into a TAMPire Scene + goal predicates.

Positions are nominal (kitchen world coordinates are far outside the tabletop
reach model); the SYMBOLIC plan reasons over predicates, not metric kitchen poses.
Execution (Milestone B) would substitute the env's real poses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .. import llm
from ..schemas import Predicate, Scene, WorldObject

ROBOCASA_PERCEPTION_SYS = """You are the PERCEPTION agent of a kitchen manipulation robot.
You see one camera frame of a kitchen counter. Using the image (and the instruction as
context), identify exactly three things:
  1. object  : the single small GRASPABLE item the robot must manipulate (on the counter)
  2. source  : the surface it currently rests on (almost always the counter)
  3. target  : the receptacle the instruction says to put the object into/onto
               (a cabinet, drawer, microwave, fridge, sink, or the counter)
For the target, judge whether it is an openable container and whether it currently
looks CLOSED (cabinets/drawers/microwaves/fridges are openable and usually start closed;
a sink or open counter is not openable).

Return ONLY JSON:
{"object": {"name": "<short noun>", "color": "<word or null>"},
 "source": {"name": "counter"},
 "target": {"name": "<cabinet|drawer|microwave|fridge|sink|counter>",
            "container": true, "openable": true, "closed": true},
 "notes": "one short sentence about what you see"}"""

ROBOCASA_PERCEPTION_USER = """Task instruction: "{instruction}"
Identify the object to manipulate, the surface it is on, and the target receptacle."""


@dataclass
class PerceivedTask:
    scene: Scene
    goal_predicates: List[Predicate]
    object_id: str
    target_id: str
    target_closed: bool
    notes: str


def _slug(name: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or fallback


def perceive(image_path: str, instruction: str) -> PerceivedTask:
    messages = [
        llm.sys(ROBOCASA_PERCEPTION_SYS),
        llm.user_with_image(
            ROBOCASA_PERCEPTION_USER.format(instruction=instruction), image_path),
    ]
    data, _ = llm.chat_json(messages, label="rc_perception", temperature=0.1)

    obj = data.get("object", {}) or {}
    tgt = data.get("target", {}) or {}
    src_name = _slug((data.get("source", {}) or {}).get("name", "counter"), "counter")

    obj_id = _slug(obj.get("name", "object"), "object")
    color = obj.get("color")
    if isinstance(color, str) and color.lower() not in ("null", "none", ""):
        obj_id = f"{_slug(color, '')}_{obj_id}".strip("_")

    target_id = _slug(tgt.get("name", "cabinet"), "cabinet")
    is_container = bool(tgt.get("container", True))
    is_openable = bool(tgt.get("openable", target_id in
                               ("cabinet", "drawer", "microwave", "fridge")))
    is_closed = bool(tgt.get("closed", is_openable))

    # --- build a symbolic scene with nominal (reachable) positions ---
    objs = [
        WorldObject(obj_id, "object", color=color if isinstance(color, str) else None,
                    position=(0.0, -0.20, 0.05), size=(0.06, 0.06, 0.04),
                    affordances=["graspable"]),
        WorldObject(src_name, "counter", position=(0.0, -0.20, 0.0),
                    size=(0.5, 0.25, 0.02), affordances=["support"]),
    ]
    tgt_aff = (["container"] if is_container else ["support"]) + (
        ["openable"] if is_openable else [])
    objs.append(WorldObject(target_id, target_id, position=(0.20, -0.18, 0.10),
                            size=(0.3, 0.3, 0.3), affordances=tgt_aff))

    preds = [
        Predicate("on", [obj_id, src_name]),
        Predicate("graspable", [obj_id]),
        Predicate("clear", [obj_id]),
    ]
    if is_container:
        preds.append(Predicate("container", [target_id]))
    if is_openable:
        preds.append(Predicate("closed" if is_closed else "open", [target_id]))

    scene = Scene(objects=objs, predicates=preds,
                  notes=f"robocasa perception: {data.get('notes','')}")
    rel = "in" if is_container else "on"
    goal_predicates = [Predicate(rel, [obj_id, target_id])]
    return PerceivedTask(scene=scene, goal_predicates=goal_predicates,
                         object_id=obj_id, target_id=target_id,
                         target_closed=is_openable and is_closed,
                         notes=str(data.get("notes", "")))
