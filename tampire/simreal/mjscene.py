"""Build a MuJoCo MJCF model from a TAMPire Scene.

Conventions:
  - table top surface is at z = 0; objects rest on it.
  - blocks are free boxes; bowls are static open trays (floor + 4 short walls).
  - object ids map 1:1 to MuJoCo body names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..schemas import Scene, WorldObject

_RGBA = {
    "red": "0.85 0.18 0.18 1", "green": "0.17 0.62 0.29 1", "blue": "0.19 0.40 0.84 1",
    "yellow": "0.95 0.76 0.05 1", "orange": "0.91 0.45 0.10 1",
    "purple": "0.56 0.27 0.68 1", "cyan": "0.10 0.71 0.77 1",
}


def _rgba(color: str) -> str:
    return _RGBA.get((color or "").lower(), "0.6 0.63 0.65 1")


@dataclass
class SceneMeta:
    """Geometry the env needs that isn't in the MJCF by name."""
    block_half: Dict[str, float] = field(default_factory=dict)       # id -> half height (z)
    bowl_inner: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)  # id -> xmin,ymin,xmax,ymax
    bowl_wall_top: Dict[str, float] = field(default_factory=dict)    # id -> wall top z
    is_block: Dict[str, bool] = field(default_factory=dict)


def build(scene: Scene) -> Tuple[str, SceneMeta]:
    meta = SceneMeta()
    body_xml: List[str] = []

    for o in scene.objects:
        is_container = o.category in ("bowl", "cup", "plate", "tray") or "container" in o.affordances
        meta.is_block[o.id] = not is_container
        x, y = o.position[0], o.position[1]
        if is_container:
            body_xml.append(_bowl_xml(o, x, y, meta))
        else:
            body_xml.append(_block_xml(o, x, y, meta))

    xmin, ymin, xmax, ymax = scene.table_bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    hx, hy = (xmax - xmin) / 2 + 0.06, (ymax - ymin) / 2 + 0.06

    xml = f"""
<mujoco model="tampire_tabletop">
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.7 0.58 0.4" rgb2="0.62 0.5 0.34"
             width="300" height="300"/>
    <material name="tablemat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.4 1.2" dir="-0.1 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="table" type="box" pos="{cx} {cy} -0.02" size="{hx} {hy} 0.02"
          material="tablemat"/>
    <body name="tabletarget" pos="{cx} {cy} 0.03">
      <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
    <camera name="topdown" pos="{cx} {cy-0.05} 0.95" xyaxes="1 0 0 0 1 0"/>
    <camera name="angled" pos="{cx} {cy-0.75} 0.6" xyaxes="1 0 0 0 0.6 0.8"/>
    <camera name="angled_left" pos="{cx-0.55} {cy-0.6} 0.55" target="tabletarget" mode="targetbody"/>
    <camera name="angled_right" pos="{cx+0.55} {cy-0.6} 0.55" target="tabletarget" mode="targetbody"/>
{''.join(body_xml)}
  </worldbody>
</mujoco>
""".strip()
    return xml, meta


def _block_xml(o: WorldObject, x: float, y: float, meta: SceneMeta) -> str:
    half = 0.025
    meta.block_half[o.id] = half
    z = o.position[2] if o.position[2] > half else half
    return f"""
    <body name="{o.id}" pos="{x} {y} {z}">
      <freejoint/>
      <geom type="box" size="{half} {half} {half}" rgba="{_rgba(o.color)}"
            mass="0.05" friction="1 0.05 0.001"/>
    </body>"""


def _bowl_xml(o: WorldObject, x: float, y: float, meta: SceneMeta) -> str:
    inner = 0.055     # inner half-extent
    wall_t = 0.006    # wall thickness
    wall_h = 0.030    # wall height
    floor_h = 0.004
    meta.bowl_inner[o.id] = (x - inner, y - inner, x + inner, y + inner)
    meta.bowl_wall_top[o.id] = 2 * wall_h
    rgba = _rgba(o.color)
    a = inner + wall_t
    return f"""
    <body name="{o.id}" pos="{x} {y} 0">
      <geom name="{o.id}_floor" type="box" pos="0 0 {floor_h}" size="{inner} {inner} {floor_h}" rgba="{rgba}"/>
      <geom name="{o.id}_w0" type="box" pos="{a} 0 {wall_h}" size="{wall_t} {a} {wall_h}" rgba="{rgba}"/>
      <geom name="{o.id}_w1" type="box" pos="-{a} 0 {wall_h}" size="{wall_t} {a} {wall_h}" rgba="{rgba}"/>
      <geom name="{o.id}_w2" type="box" pos="0 {a} {wall_h}" size="{a} {wall_t} {wall_h}" rgba="{rgba}"/>
      <geom name="{o.id}_w3" type="box" pos="0 -{a} {wall_h}" size="{a} {wall_t} {wall_h}" rgba="{rgba}"/>
    </body>"""
