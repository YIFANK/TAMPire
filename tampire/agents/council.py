"""Debate council: given a failed plan + simulator verdict, several critics
diagnose the failure through distinct lenses, then a repair chair synthesizes a
single corrected plan.

This is the part that only works because Cerebras is fast: N critics + a repair
agent per round, multiple rounds, in real time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .. import llm, prompts
from ..config import CONFIG
from ..schemas import FeasibilityResult, Plan, Scene


@dataclass
class Diagnosis:
    persona: str
    text: str


# stream callback: (event_type, who, delta_text)
#   event_type in {"critic_start","critic_delta","critic_end","repair_start","repair_delta","repair_end"}
StreamCB = Optional[Callable[[str, str, str], None]]


def _format_common(scene: Scene, goal: str, plan: Plan, verdict: FeasibilityResult):
    predicates = "\n".join(f"  {p}" for p in scene.predicates) or "  (none)"
    plan_str = plan.pretty()
    verdict_str = verdict.summary() + "\n" + "\n".join(
        f"    step {c.index+1}: {'OK' if c.ok else 'FAIL — ' + c.reason}" for c in verdict.checks
    )
    return predicates, plan_str, verdict_str


def run_critics(
    scene: Scene, goal: str, plan: Plan, verdict: FeasibilityResult,
    *, stream_cb: StreamCB = None,
) -> List[Diagnosis]:
    predicates, plan_str, verdict_str = _format_common(scene, goal, plan, verdict)
    personas = prompts.CRITIC_PERSONAS[: CONFIG.council_size]
    diagnoses: List[Diagnosis] = []

    for persona in personas:
        messages = [
            llm.sys(prompts.CRITIC_SYS.format(persona=persona)),
            llm.user(prompts.CRITIC_USER.format(
                goal=goal, predicates=predicates, plan=plan_str, verdict=verdict_str)),
        ]
        if stream_cb:
            stream_cb("critic_start", persona, "")
            buf: List[str] = []
            for delta in llm.stream(messages, label="critic", temperature=0.5, max_tokens=300):
                buf.append(delta)
                stream_cb("critic_delta", persona, delta)
            text = "".join(buf)
            stream_cb("critic_end", persona, text)
        else:
            text, _ = llm.chat(messages, label="critic", temperature=0.5, max_tokens=300)
        diagnoses.append(Diagnosis(persona=persona, text=text.strip()))

    return diagnoses


def repair(
    scene: Scene, goal: str, plan: Plan, verdict: FeasibilityResult,
    diagnoses: List[Diagnosis], *, stream_cb: StreamCB = None,
) -> Plan:
    predicates, plan_str, verdict_str = _format_common(scene, goal, plan, verdict)
    diag_str = "\n".join(
        f"  - ({d.persona.split('—')[0].strip()}): {d.text}" for d in diagnoses
    )
    messages = [
        llm.sys(prompts.REPAIR_SYS),
        llm.user(prompts.REPAIR_USER.format(
            goal=goal, predicates=predicates, plan=plan_str,
            verdict=verdict_str, diagnoses=diag_str)),
    ]
    if stream_cb:
        stream_cb("repair_start", "repair", "")
    data, _ = llm.chat_json(messages, label="repair", temperature=0.3, max_tokens=3000)
    new_plan = Plan.from_dict(data)
    if stream_cb:
        stream_cb("repair_end", "repair", new_plan.rationale)
    return new_plan


def debate_and_repair(
    scene: Scene, goal: str, plan: Plan, verdict: FeasibilityResult,
    *, stream_cb: StreamCB = None,
) -> Plan:
    diagnoses = run_critics(scene, goal, plan, verdict, stream_cb=stream_cb)
    return repair(scene, goal, plan, verdict, diagnoses, stream_cb=stream_cb)
