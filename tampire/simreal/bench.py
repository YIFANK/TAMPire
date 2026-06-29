"""Tier-1 PHYSICS benchmark over the full pixel -> plan -> physics loop.

Unlike Tier 0 (which planned over ground-truth symbolic scenes), this renders each
scene, runs perception on the REAL pixels, plans, executes in MuJoCo, and scores
success from physics. So it also measures the perception/grounding error that Tier 0
never saw — a strictly harder, more honest number.

  python -m tampire.simreal.bench --n 10 --conditions baseline council
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from ..eval import scenegen
from ..llm import METRICS
from .runner import run_from_pixels

console = Console()


@dataclass
class Row:
    seed: int
    kind: str
    difficulty: str
    condition: str
    symbolic: bool
    physics: bool
    agree: bool
    repairs: int
    wall_s: float


def _agg(rows: List[Row]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "physics_succ": sum(r.physics for r in rows) / n,
        "symbolic_succ": sum(r.symbolic for r in rows) / n,
        "agreement": sum(r.agree for r in rows) / n,
        "avg_repairs": sum(r.repairs for r in rows) / n,
        "avg_wall": sum(r.wall_s for r in rows) / n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TAMPire Tier-1 physics benchmark")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "council"],
                    choices=["baseline", "council"])
    ap.add_argument("--camera", default="angled", choices=["angled", "topdown"])
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--tmp", default="runs/bench_tmp")
    args = ap.parse_args(argv)

    os.makedirs(args.tmp, exist_ok=True)
    console.print(f"[bold]TAMPire Tier-1 physics benchmark[/bold]  ·  {args.n} scenes  ·  "
                  f"{', '.join(args.conditions)}\n[dim]plans from real pixels; "
                  f"success scored in physics[/dim]\n")

    rows: Dict[str, List[Row]] = {c: [] for c in args.conditions}
    for i in range(args.n):
        seed = args.seed + i
        task = scenegen.generate(seed)
        for cond in args.conditions:
            METRICS.reset()
            # baseline = myopic planner + NO council (0 repair rounds), to isolate
            # the council's contribution; council = full planner + repair council.
            res = run_from_pixels(
                task.scene, task.goal, task.goal_predicates,
                camera=args.camera, baseline=(cond == "baseline"),
                max_repair_rounds=(0 if cond == "baseline" else None),
                out_prefix=f"{args.tmp}/s{seed}_{cond}", save_frames=False,
            )
            row = Row(seed, task.kind, task.difficulty, cond,
                      res.symbolic_success, res.physics_success,
                      res.symbolic_success == res.physics_success,
                      res.repairs, METRICS.total_wall_s)
            rows[cond].append(row)
            p = "[green]✓[/green]" if res.physics_success else "[red]✗[/red]"
            console.print(f"  seed {seed} [{task.difficulty:>7}/{task.kind:>7}] "
                          f"{cond:<8} physics {p} (sym {'✓' if res.symbolic_success else '✗'}, "
                          f"{res.repairs} rep)", highlight=False)

    table = Table(title="\nTier-1 physics benchmark (from real pixels)", title_style="bold")
    table.add_column("condition", style="bold")
    table.add_column("physics succ", justify="right")
    table.add_column("symbolic succ", justify="right")
    table.add_column("agreement", justify="right")
    table.add_column("avg repairs", justify="right")
    table.add_column("avg wall", justify="right")
    for c in args.conditions:
        a = _agg(rows[c])
        if not a.get("n"):
            continue
        col = "green" if a["physics_succ"] >= 0.8 else ("yellow" if a["physics_succ"] >= 0.5 else "red")
        table.add_row(c, f"[{col}]{a['physics_succ']*100:.0f}%[/{col}]",
                      f"{a['symbolic_succ']*100:.0f}%", f"{a['agreement']*100:.0f}%",
                      f"{a['avg_repairs']:.2f}", f"{a['avg_wall']:.2f}s")
    console.print(table)

    if "baseline" in rows and "council" in rows:
        b = _agg(rows["baseline"]).get("physics_succ", 0)
        c = _agg(rows["council"]).get("physics_succ", 0)
        console.print(f"\n[bold]Council vs baseline in physics: "
                      f"{b*100:.0f}% → {c*100:.0f}%[/bold]")
    console.print("[dim]Note: physics < Tier-0 symbolic is expected — this also pays "
                  "the perception/grounding error tax of planning from real pixels.[/dim]")

    if args.json:
        payload = {"n": args.n, "seed": args.seed,
                   "raw": {c: [r.__dict__ for r in rs] for c, rs in rows.items()},
                   "aggregate": {c: _agg(rs) for c, rs in rows.items()}}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
