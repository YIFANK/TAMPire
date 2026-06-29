"""Top-down renderer for the tabletop world.

Draws the table, the robot base + reach circle, and every object from a `World`
state. Stacks (z) are faked with a small up-right offset so the viewer can see
"green on red". Can render a single state, or step a plan and emit per-frame PNGs
plus an animated GIF — the artifact judges actually look at.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from PIL import Image

from .schemas import Plan, Predicate, Scene
from .sim import feasibility
from .sim.world import BASE_XY, REACH_RADIUS, World

# named color -> RGB; fall back to grey
_COLORS = {
    "red": "#d62828", "green": "#2a9d4a", "blue": "#3066d6",
    "yellow": "#f4c20d", "orange": "#e8731a", "purple": "#8e44ad",
    "cyan": "#1ab6c4", "white": "#eaeaea", "black": "#333333",
}
_STACK_DX, _STACK_DY = 0.022, 0.022  # per-level iso offset (metres)
_DISPLAY_SCALE = 1.9                 # enlarge block glyphs for legibility (true size is tiny)


def _color(obj) -> str:
    return _COLORS.get((obj.color or "").lower(), "#9aa0a6")


def _stack_level(world: World, oid: str) -> int:
    """How many objects this one is stacked above (0 = on table/in bowl)."""
    lvl, cur, seen = 0, oid, set()
    while cur in world.on and world.on[cur] != "table" and cur not in seen:
        seen.add(cur)
        cur = world.on[cur]
        lvl += 1
    return lvl


def draw_world(ax, world: World, title: str = "", highlight: Optional[str] = None,
               title_color: str = "#222") -> None:
    xmin, ymin, xmax, ymax = world.table_bounds
    pad = 0.08
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(BASE_XY[1] - 0.08, ymax + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", family="monospace",
                     color=title_color)

    # table
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor="#c49e6e", edgecolor="#8a6a3a", lw=2, zorder=0))
    # reach circle + base
    ax.add_patch(Circle(BASE_XY, REACH_RADIUS, facecolor="none",
                        edgecolor="#888", ls="--", lw=1, zorder=1))
    ax.add_patch(Circle(BASE_XY, 0.03, facecolor="#222", edgecolor="none", zorder=5))
    ax.annotate("robot", BASE_XY, textcoords="offset points", xytext=(0, -12),
                ha="center", fontsize=7, color="#444")

    # objects, drawn lowest-stack first so higher ones overlap on top
    order = sorted(world.objects.values(), key=lambda o: _stack_level(world, o.id))
    for o in order:
        if world.holding == o.id:
            # held: park it at the gripper (base), with a marker
            cx, cy = BASE_XY[0], BASE_XY[1] + 0.06
            _draw_obj_glyph(ax, o, cx, cy, held=True)
            continue
        if o.id in world.inside:
            container = world.objects.get(world.inside[o.id])
            cx, cy = (container.position[0], container.position[1]) if container else o.position[:2]
            _draw_obj_glyph(ax, o, cx, cy, scale=0.6)
            continue
        lvl = _stack_level(world, o.id)
        # a stacked object sits on its support's x,y, nudged by level
        if lvl > 0 and o.id in world.on:
            base = world.objects.get(world.on[o.id])
            bx, by = (base.position[0], base.position[1]) if base else o.position[:2]
            cx, cy = bx + lvl * _STACK_DX, by + lvl * _STACK_DY
        else:
            cx, cy = o.position[0], o.position[1]
        _draw_obj_glyph(ax, o, cx, cy, highlight=(o.id == highlight))


def _draw_obj_glyph(ax, o, cx, cy, scale=1.0, held=False, highlight=False) -> None:
    is_container = o.category in ("bowl", "cup", "plate", "tray") or "container" in o.affordances
    disp = scale if is_container else scale * _DISPLAY_SCALE
    sx = max(o.size[0], 0.03) * disp
    sy = max(o.size[1], 0.03) * disp
    edge = "#111" if not highlight else "#ff2d55"
    lw = 1.5 if not highlight else 3.0
    z = 8 if held else 4
    if is_container:
        ax.add_patch(Ellipse((cx, cy), sx, sy, facecolor=_color(o),
                             edgecolor=edge, lw=lw, alpha=0.85, zorder=z))
    else:
        ax.add_patch(FancyBboxPatch((cx - sx / 2, cy - sy / 2), sx, sy,
                     boxstyle="round,pad=0.002,rounding_size=0.01",
                     facecolor=_color(o), edgecolor=edge, lw=lw, zorder=z))
    label = o.id + ("  (held)" if held else "")
    ax.annotate(label, (cx, cy + sy * 0.5 + 0.012), ha="center", va="bottom",
                fontsize=6.5, color="#111", fontweight="bold", zorder=z + 1)


def render_state(world: World, path: str, title: str = "",
                 title_color: str = "#222", highlight: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=130)
    draw_world(ax, world, title, highlight=highlight, title_color=title_color)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_plan(
    scene: Scene, plan: Plan, goal_predicates: List[Predicate],
    out_prefix: str, *, gif: bool = True, fps: float = 1.2,
) -> Tuple[List[str], Optional[str]]:
    """Step the plan in a fresh World, render a frame after each step, and
    assemble a GIF. Returns (frame_paths, gif_path|None)."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    world = World.from_scene(scene)
    frames: List[str] = []

    # frame 0: initial state
    frames.append(render_state(world, f"{out_prefix}_00.png", "initial state"))

    for i, step in enumerate(plan.steps, 1):
        ok, reason = world.apply(step)
        hl = step.args[0] if step.args else None
        tag = f"{i}. {step}" + ("" if ok else f"   FAIL: {reason}")
        color = "#222" if ok else "#c0392b"
        frames.append(render_state(world, f"{out_prefix}_{i:02d}.png", tag,
                                   title_color=color, highlight=hl))
        if not ok:
            break

    # final caption
    verdict = feasibility.check(scene, plan, goal_predicates)
    if verdict.ok and verdict.goal_satisfied:
        cap, color = "VERIFIED  -  goal satisfied", "#1e8449"
    else:
        cap, color = verdict.summary(), "#c0392b"
    frames.append(render_state(world, f"{out_prefix}_final.png", cap, title_color=color))

    gif_path = None
    if gif and frames:
        imgs = [Image.open(f).convert("RGB") for f in frames]
        w = max(im.width for im in imgs); h = max(im.height for im in imgs)
        imgs = [im.resize((w, h)) for im in imgs]
        gif_path = f"{out_prefix}.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
    return frames, gif_path
