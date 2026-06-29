"""Run the Tier-0 ablation and print a results table.

  python -m tampire.eval --n 12
  python -m tampire.eval --n 24 --conditions baseline council --json runs/eval.json
"""
from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from . import harness

console = Console()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TAMPire Tier-0 evaluation")
    ap.add_argument("--n", type=int, default=12, help="number of generated scenes")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--conditions", nargs="+", default=list(harness.CONDITIONS),
                    help=f"subset of {harness.CONDITIONS}")
    ap.add_argument("--json", help="write raw + aggregate results to this path")
    args = ap.parse_args(argv)

    console.print(f"[bold]TAMPire Tier-0 eval[/bold]  ·  {args.n} scenes  ·  "
                  f"conditions: {', '.join(args.conditions)}\n")

    def progress(i, n, task, rec):
        mark = "[green]✓[/green]" if rec.success else "[red]✗[/red]"
        console.print(f"  scene {i+1:>2}/{n} [{task.difficulty:>7}/{task.kind:>7}] "
                      f"{rec.condition:<8} {mark}  ({rec.rounds} rep, {rec.wall_s:.2f}s)",
                      highlight=False)

    results = harness.run_suite(args.n, conditions=tuple(args.conditions),
                                base_seed=args.seed, progress=progress)

    table = Table(title="\nAblation — success rate & cost", title_style="bold")
    table.add_column("condition", style="bold")
    table.add_column("success", justify="right")
    table.add_column("avg repairs", justify="right")
    table.add_column("avg wall", justify="right")
    table.add_column("avg tokens", justify="right")
    for c in args.conditions:
        agg = harness.aggregate(results[c])
        sr = agg.success_rate
        color = "green" if sr >= 0.9 else ("yellow" if sr >= 0.5 else "red")
        table.add_row(c, f"[{color}]{sr*100:.0f}%[/{color}]  ({agg.successes}/{agg.n})",
                      f"{agg.avg_rounds:.2f}", f"{agg.avg_wall:.2f}s", f"{agg.avg_tokens:.0f}")
    console.print(table)

    # headline delta
    if "baseline" in results and "council" in results:
        b = harness.aggregate(results["baseline"]).success_rate
        c = harness.aggregate(results["council"]).success_rate
        console.print(f"\n[bold]Debate council lifts success "
                      f"{b*100:.0f}% → {c*100:.0f}%[/bold] "
                      f"(+{(c-b)*100:.0f} pts) by repairing infeasible plans.")

    if args.json:
        payload = {
            "n": args.n, "seed": args.seed,
            "raw": {c: [r.__dict__ for r in recs] for c, recs in results.items()},
            "aggregate": {c: harness.aggregate(recs).__dict__ for c, recs in results.items()},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
