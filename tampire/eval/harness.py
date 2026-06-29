"""Ablation harness: run conditions over the procedural suite and aggregate.

Conditions
  oracle   : deterministic solver (sanity; should be 100%)
  baseline : myopic LLM planner, NO repair council  (0 rounds)
  council  : myopic LLM planner + debate repair council
  smart    : full LLM planner + debate repair council

The headline result: council recovers most of baseline's failures, and does it
fast (Cerebras). All success checks come from the deterministic verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .. import pipeline
from ..llm import METRICS
from ..sim import feasibility
from . import oracle, scenegen


@dataclass
class RunRecord:
    seed_idx: int
    difficulty: str
    kind: str
    condition: str
    success: bool
    rounds: int          # repair rounds used (0 = solved on first plan)
    wall_s: float
    model_ms: float
    tokens: int
    calls: int


@dataclass
class Aggregate:
    condition: str
    n: int = 0
    successes: int = 0
    rounds_sum: int = 0
    wall_sum: float = 0.0
    model_ms_sum: float = 0.0
    tokens_sum: int = 0

    def add(self, r: RunRecord) -> None:
        self.n += 1
        self.successes += int(r.success)
        self.rounds_sum += r.rounds
        self.wall_sum += r.wall_s
        self.model_ms_sum += r.model_ms
        self.tokens_sum += r.tokens

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def avg_rounds(self) -> float:
        return self.rounds_sum / self.n if self.n else 0.0

    @property
    def avg_wall(self) -> float:
        return self.wall_sum / self.n if self.n else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens_sum / self.n if self.n else 0.0


CONDITIONS = ("oracle", "baseline", "council", "smart")


def _run_condition(task: scenegen.EvalTask, condition: str, idx: int) -> RunRecord:
    METRICS.reset()
    if condition == "oracle":
        plan = oracle.solve(task)
        v = feasibility.check(task.scene, plan, task.goal_predicates)
        success, rounds = (v.ok and v.goal_satisfied), 0
    else:
        myopic = condition in ("baseline", "council")
        rounds_budget = 0 if condition == "baseline" else None  # None -> CONFIG default
        res = pipeline.run(
            task.goal,
            scene=task.scene,
            goal_predicates=task.goal_predicates,
            myopic_planner=myopic,
            max_repair_rounds=rounds_budget,
        )
        success = res.success
        rounds = len(res.rounds) - 1
    return RunRecord(
        seed_idx=idx, difficulty=task.difficulty, kind=task.kind, condition=condition,
        success=success, rounds=rounds,
        wall_s=METRICS.total_wall_s, model_ms=METRICS.total_model_s * 1000,
        tokens=METRICS.total_tokens, calls=METRICS.n,
    )


def run_suite(n: int, conditions=CONDITIONS, base_seed: int = 1000,
              progress=None) -> Dict[str, List[RunRecord]]:
    tasks = scenegen.generate_suite(n, base_seed=base_seed)
    out: Dict[str, List[RunRecord]] = {c: [] for c in conditions}
    for i, task in enumerate(tasks):
        for c in conditions:
            rec = _run_condition(task, c, i)
            out[c].append(rec)
            if progress:
                progress(i, n, task, rec)
    return out


def aggregate(records: List[RunRecord]) -> Aggregate:
    agg = Aggregate(condition=records[0].condition if records else "")
    for r in records:
        agg.add(r)
    return agg
