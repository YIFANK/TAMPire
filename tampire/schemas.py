"""Core data structures shared across agents and the simulator.

Everything is a plain dataclass with to_dict/from_dict so it serializes cleanly
into LLM prompts and back out of JSON responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


def _fmt_num(x: Any) -> str:
    """Stringify a predicate/step arg that came back as a number."""
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)


# ----------------------------------------------------------------------------
# Perception / scene
# ----------------------------------------------------------------------------
@dataclass
class WorldObject:
    """A physical object detected in the scene."""
    id: str                       # stable symbol, e.g. "red_block"
    category: str                 # "block", "bowl", "table", ...
    color: Optional[str] = None
    # 2.5D pose on the tabletop. x,y in metres on the table plane, z = height of top.
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04)  # bbox extents (m)
    affordances: List[str] = field(default_factory=list)   # ["graspable", "container", ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldObject":
        return cls(
            id=d["id"],
            category=d.get("category", "object"),
            color=d.get("color"),
            position=tuple(d.get("position", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            size=tuple(d.get("size", (0.04, 0.04, 0.04))),       # type: ignore[arg-type]
            affordances=list(d.get("affordances", [])),
        )


@dataclass
class Predicate:
    """A symbolic relation, PDDL-style: name + ordered args, optionally negated."""
    name: str                     # "on", "clear", "in", "graspable", "at"
    args: List[str] = field(default_factory=list)
    negated: bool = False

    def key(self) -> Tuple[str, Tuple[str, ...], bool]:
        return (self.name, tuple(self.args), self.negated)

    def __str__(self) -> str:
        body = f"{self.name}({', '.join(self.args)})"
        return f"not {body}" if self.negated else body

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": list(self.args), "negated": self.negated}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Predicate":
        # args may come back as numbers (e.g. at(obj, 0.1, 0.05)); normalize to str
        args = [a if isinstance(a, str) else _fmt_num(a) for a in d.get("args", [])]
        return cls(name=d["name"], args=args, negated=bool(d.get("negated", False)))


@dataclass
class Scene:
    objects: List[WorldObject] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    table_bounds: Tuple[float, float, float, float] = (-0.4, -0.3, 0.4, 0.3)  # xmin,ymin,xmax,ymax
    notes: str = ""

    def by_id(self, oid: str) -> Optional[WorldObject]:
        return next((o for o in self.objects if o.id == oid), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "predicates": [p.to_dict() for p in self.predicates],
            "table_bounds": list(self.table_bounds),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scene":
        return cls(
            objects=[WorldObject.from_dict(o) for o in d.get("objects", [])],
            predicates=[Predicate.from_dict(p) for p in d.get("predicates", [])],
            table_bounds=tuple(d.get("table_bounds", (-0.4, -0.3, 0.4, 0.3))),  # type: ignore[arg-type]
            notes=d.get("notes", ""),
        )


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
@dataclass
class PlanStep:
    """A high-level action. Maps 1:1 to a robot primitive at execution time."""
    action: str                   # "pick", "place", "move_to", "open_gripper"...
    args: List[str] = field(default_factory=list)
    rationale: str = ""

    def __str__(self) -> str:
        return f"{self.action}({', '.join(self.args)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "args": list(self.args), "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanStep":
        args = [a if isinstance(a, str) else _fmt_num(a) for a in d.get("args", [])]
        return cls(action=d["action"], args=args, rationale=d.get("rationale", ""))


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            rationale=d.get("rationale", ""),
        )

    def pretty(self) -> str:
        return "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps)) or "  (empty plan)"


# ----------------------------------------------------------------------------
# Feasibility / verification
# ----------------------------------------------------------------------------
@dataclass
class StepCheck:
    index: int                    # which plan step (0-based)
    ok: bool
    reason: str = ""              # human + LLM readable failure reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityResult:
    ok: bool
    checks: List[StepCheck] = field(default_factory=list)
    goal_satisfied: bool = False

    def first_failure(self) -> Optional[StepCheck]:
        return next((c for c in self.checks if not c.ok), None)

    def failures(self) -> List[StepCheck]:
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        if self.ok and self.goal_satisfied:
            return "FEASIBLE and goal satisfied."
        if self.ok and not self.goal_satisfied:
            return "All steps feasible, but goal NOT satisfied at the end."
        f = self.first_failure()
        return f"INFEASIBLE at step {f.index + 1}: {f.reason}" if f else "INFEASIBLE."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "goal_satisfied": self.goal_satisfied,
            "checks": [c.to_dict() for c in self.checks],
        }
