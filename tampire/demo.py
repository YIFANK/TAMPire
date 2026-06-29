"""TAMPire live demo CLI.

  python -m tampire.demo --scene scenes/blocks.json --goal "put the red block in the bowl"
  python -m tampire.demo --image scenes/scene.png  --goal "stack the blue block on the red block"

Renders each stage, streams the debate council live, and shows a latency panel —
the "speed is the architecture" money shot.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from . import pipeline
from .llm import METRICS
from .schemas import FeasibilityResult, Plan

console = Console()


def _verdict_text(v: FeasibilityResult) -> str:
    color = "green" if (v.ok and v.goal_satisfied) else "red"
    return f"[{color}]{v.summary()}[/{color}]"


def make_event_handler():
    def on_event(stage: str, payload: Dict[str, Any]) -> None:
        if stage == "perception_start":
            console.print(Rule("[bold cyan]1 · Perception (pixels → objects)"))
            console.print(f"  reading [italic]{payload['image']}[/italic] …")
        elif stage == "perception_done":
            scene = payload["scene"]
            objs = ", ".join(f"[bold]{o.id}[/bold]" for o in scene.objects)
            console.print(f"  detected: {objs}")
            if scene.notes:
                console.print(f"  [dim]{scene.notes}[/dim]")
        elif stage == "grounding_start":
            console.print(Rule("[bold cyan]2 · Grounding (objects → predicates)"))
        elif stage == "grounding_done":
            scene = payload["scene"]
            for p in scene.predicates:
                console.print(f"  • {p}")
        elif stage == "goalspec_done":
            gp = ", ".join(str(p) for p in payload["goal_predicates"]) or "(none)"
            console.print(Rule("[bold cyan]3 · Goal spec"))
            console.print(f"  target: [bold yellow]{gp}[/bold yellow]")
        elif stage == "plan_done":
            console.print(Rule("[bold cyan]4 · Initial plan"))
            console.print(payload["plan"].pretty())
            console.print(f"  verdict → {_verdict_text(payload['verdict'])}")
        elif stage == "debate_start":
            r = payload["round"]
            console.print(Rule(f"[bold magenta]Debate round {r} · council convenes"))
            f = payload["verdict"].first_failure()
            if f:
                console.print(f"  [red]failure to fix:[/red] step {f.index+1} — {f.reason}\n")
        elif stage == "repair_done":
            console.print()
            console.print(f"  [bold]repaired plan (round {payload['round']}):[/bold]")
            console.print(payload["plan"].pretty())
            console.print(f"  verdict → {_verdict_text(payload['verdict'])}")
        elif stage == "success":
            console.print(Rule("[bold green]✅ VERIFIED"))
        elif stage == "failure":
            console.print(Rule("[bold red]✗ budget exhausted — best effort plan"))

    return on_event


def make_council_streamer():
    """Streams critics' reasoning token-by-token (plain stdout to avoid markup clashes)."""
    state = {"persona": ""}

    def on_stream(event: str, who: str, delta: str) -> None:
        if event == "critic_start":
            lens = who.split("—")[0].strip()
            state["persona"] = lens
            console.print(f"\n  [bold]🗣  critic · {lens}[/bold]")
            sys.stdout.write("    ")
            sys.stdout.flush()
        elif event == "critic_delta":
            sys.stdout.write(delta)
            sys.stdout.flush()
        elif event == "critic_end":
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif event == "repair_start":
            console.print("\n  [bold]🔧 chair synthesizes a fix …[/bold]")
        elif event == "repair_end":
            if who and delta:
                console.print(f"    [dim]{delta}[/dim]")

    return on_stream


def latency_panel() -> Panel:
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("metric"); t.add_column("value", justify="right")
    t.add_row("LLM calls", str(METRICS.n))
    t.add_row("total tokens", str(METRICS.total_tokens))
    t.add_row("model compute", f"{METRICS.total_model_s*1000:.0f} ms")
    t.add_row("wall clock (incl. net)", f"{METRICS.total_wall_s:.2f} s")
    t.add_row("throughput", f"{METRICS.tokens_per_s:.0f} tok/s")
    return Panel(t, title="[bold]⚡ Cerebras speed[/bold]", border_style="yellow", expand=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TAMPire live demo")
    ap.add_argument("--goal", required=True, help="natural-language goal")
    ap.add_argument("--scene", help="path to a JSON scene")
    ap.add_argument("--image", help="path to a scene image (runs perception)")
    ap.add_argument("--baseline", action="store_true",
                    help="use the myopic baseline planner to reliably trigger the repair council")
    ap.add_argument("--render", metavar="PREFIX", nargs="?", const="runs/plan",
                    help="render the final verified plan top-down to PREFIX_*.png + PREFIX.gif")
    args = ap.parse_args(argv)

    if not args.scene and not args.image:
        ap.error("provide --scene or --image")

    METRICS.reset()
    console.print(Panel.fit(
        f"[bold]TAMPire[/bold] 🔥  goal: [yellow]{args.goal}[/yellow]",
        border_style="cyan"))

    result = pipeline.run(
        args.goal,
        scene_path=args.scene,
        image_path=args.image,
        events=make_event_handler(),
        council_stream=make_council_streamer(),
        myopic_planner=args.baseline,
    )

    console.print(Rule("[bold]Robot-executable primitive sequence"))
    if result.final_plan and result.final_plan.steps:
        for i, p in enumerate(result.primitives, 1):
            argstr = ", ".join(p["args"])
            console.print(f"  {i:>2}. [bold green]{p['action']}[/bold green]({argstr})")
    else:
        console.print("  (no plan produced)")

    if args.render and result.final_plan and result.final_plan.steps:
        from . import render
        frames, gif = render.render_plan(
            result.scene, result.final_plan, result.goal_predicates, args.render)
        console.print(Rule("[bold]Top-down render"))
        console.print(f"  {len(frames)} frames → [italic]{args.render}_*.png[/italic]")
        if gif:
            console.print(f"  animation → [bold cyan]{gif}[/bold cyan]")

    console.print()
    console.print(latency_panel())
    status = "[bold green]SUCCESS[/bold green]" if result.success else "[bold red]UNSOLVED[/bold red]"
    console.print(f"\n{status} in {len(result.rounds)} round(s) "
                  f"({len(result.rounds)-1} repair(s)).")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
