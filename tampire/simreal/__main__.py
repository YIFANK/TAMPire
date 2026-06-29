"""Run TAMPire end-to-end through the MuJoCo sim, from REAL rendered pixels.

  # generated scene (ground-truth success check available)
  python -m tampire.simreal --seed 1003 --baseline

  # your own JSON scene + goal
  python -m tampire.simreal --scene scenes/blocks.json --goal "put the red block in the bowl"
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ..eval import scenegen
from ..pipeline import load_scene
from ..schemas import Predicate
from .runner import run_from_pixels

console = Console()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TAMPire Tier-1 (MuJoCo) run from pixels")
    ap.add_argument("--seed", type=int, help="use a generated scene with this seed")
    ap.add_argument("--scene", help="path to a JSON scene")
    ap.add_argument("--goal", help="goal (required with --scene)")
    ap.add_argument("--camera", default="angled", choices=["angled", "topdown"])
    ap.add_argument("--baseline", action="store_true", help="myopic planner (triggers repair)")
    ap.add_argument("--out", default="runs/mjrun")
    args = ap.parse_args(argv)

    if args.seed is not None:
        task = scenegen.generate(args.seed)
        scene, goal, gpreds = task.scene, task.goal, task.goal_predicates
        console.print(f"[dim]generated seed={args.seed} ({task.difficulty}/{task.kind})[/dim]")
    elif args.scene and args.goal:
        scene = load_scene(args.scene)
        goal = args.goal
        # no ground-truth predicates for arbitrary scenes; infer a best-effort check
        gpreds = _guess_goal_predicates(scene, goal)
    else:
        ap.error("provide --seed, or both --scene and --goal")

    console.print(Panel.fit(
        f"[bold]TAMPire · MuJoCo[/bold] 🦾  goal: [yellow]{goal}[/yellow]\n"
        f"[dim]planning from REAL rendered pixels; success checked in physics[/dim]",
        border_style="cyan"))

    res = run_from_pixels(scene, goal, gpreds, camera=args.camera,
                          baseline=args.baseline, out_prefix=args.out)

    console.print(Rule("Result"))
    console.print("  executed plan:")
    for i, s in enumerate(res.final_plan.steps, 1):
        console.print(f"    {i:>2}. [green]{s.action}[/green]({', '.join(s.args)})")
    sym = "[green]PASS[/green]" if res.symbolic_success else "[red]FAIL[/red]"
    phy = "[green]PASS[/green]" if res.physics_success else "[red]FAIL[/red]"
    console.print(f"\n  symbolic verifier : {sym}")
    console.print(f"  physics  verifier : {phy}  [dim]{res.physics_reason}[/dim]")
    console.print(f"  repairs used      : {res.repairs}")
    if res.gif:
        console.print(f"\n  real-pixel animation → [bold cyan]{res.gif}[/bold cyan]")
        console.print(f"  initial render       → {res.init_image}")
    agree = res.symbolic_success == res.physics_success
    console.print(f"\n  verifiers agree: {'[green]yes[/green]' if agree else '[red]NO — investigate[/red]'}")
    return 0 if res.physics_success else 1


def _guess_goal_predicates(scene, goal: str):
    """Cheap heuristic goal compiler for arbitrary scenes (Tier-1 has no oracle)."""
    from ..agents import goalspec
    return goalspec.compile_goal(scene, goal)


if __name__ == "__main__":
    raise SystemExit(main())
