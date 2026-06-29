"""Tier-2 CLI: TAMPire drives robosuite's Stack task from real pixels.

  python -m tampire.tier2 --seed 0
  python -m tampire.tier2 --seed 3 --baseline
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .run import run_stack

console = Console()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TAMPire x robosuite (Tier-2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="myopic planner (exercises the repair council)")
    ap.add_argument("--vision", action="store_true",
                    help="grasp from multi-view vision-estimated poses (no privileged state)")
    ap.add_argument("--out", default="runs/rs")
    args = ap.parse_args(argv)

    console.print(Panel.fit(
        "[bold]TAMPire × robosuite[/bold] 🤖  task: [yellow]Stack (Panda)[/yellow]\n"
        "[dim]perceive real agentview pixels → plan → scripted skills → "
        "robosuite success[/dim]", border_style="cyan"))

    res = run_stack(seed=args.seed, baseline=args.baseline, vision=args.vision,
                    out_prefix=args.out)

    console.print(Rule("Plan"))
    if res.plan:
        for i, s in enumerate(res.plan.steps, 1):
            console.print(f"  {i:>2}. [green]{s.action}[/green]({', '.join(s.args)})")
    console.print(Rule("Executed skills"))
    for sk in res.executed_skills:
        console.print(f"  • {sk}")

    console.print(Rule("Result"))
    sym = "[green]PASS[/green]" if res.symbolic_success else "[red]FAIL[/red]"
    rs = "[green]PASS[/green]" if res.robosuite_success else "[red]FAIL[/red]"
    console.print(f"  TAMPire symbolic verifier : {sym}")
    console.print(f"  robosuite native success  : {rs}")
    console.print(f"  repairs used              : {res.repairs}")
    if res.vision:
        pe = f"{res.vision_pose_error_cm:.1f} cm" if res.vision_pose_error_cm is not None else "n/a"
        console.print(f"  pose source               : [bold]VISION[/bold] "
                      f"(multi-view triangulation, est. error {pe})")
    if res.gif:
        console.print(f"\n  real agentview animation → [bold cyan]{res.gif}[/bold cyan]")
    return 0 if res.robosuite_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
