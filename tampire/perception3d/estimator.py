"""Multi-agent vision pose estimator.

A council of N Gemma-4 vision agents each localizes every object's base-contact
pixel and stack level. Each pixel is back-projected through the known camera onto
the appropriate support plane (z = level * block_height) to get a metric position.
Estimates are fused across agents with a robust median; the spread is the confidence.

No privileged state — this is what makes execution vision-based, and the per-agent
disagreement is exactly the multi-agent grounding signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import llm
from .camera import Camera

POSE_SYS = """You are a vision agent localizing tabletop objects for a robot.

CRITICAL: count blocks carefully. Two blocks of DIFFERENT colors stacked on each
other look like one tall two-tone tower — report BOTH as separate objects. A block
resting on another is a separate object with stack_level >= 1. Never merge a stack
into one object.

For EVERY manipulable object (every block, bowl, cup) report:
  id          : lowercase snake_case, e.g. "red_block", "blue_bowl"
  color       : dominant color word
  category    : block | bowl | cup | object
  base_xy     : [u, v] image coordinates of the object's BASE (where it contacts
                whatever is below it), each NORMALIZED to 0..1 (u=left->right,
                v=top->bottom)
  stack_level : 0 if it sits directly on the table; 1 if it sits on top of one
                block; 2 if on top of two; etc.

Return ONLY JSON: {"objects":[{...}, ...]}"""

POSE_USER = "Localize every object's base. The robot's goal will be: \"{goal}\"."


@dataclass
class PoseEstimate:
    object_id: str
    color: Optional[str]
    category: str
    xyz: np.ndarray
    xy_std_cm: float          # spread across agents = confidence (lower is better)
    n_votes: int
    stack_level: int = 0

    def to_row(self) -> str:
        return (f"{self.object_id:14} {str(np.round(self.xyz,3)):>22} "
                f"lvl={self.stack_level} votes={self.n_votes} ±{self.xy_std_cm:.1f}cm")


def _query_agent(image_path: str, goal: str, temperature: float) -> List[dict]:
    msgs = [
        llm.sys(POSE_SYS),
        llm.user_with_image(POSE_USER.format(goal=goal), image_path),
    ]
    data, _ = llm.chat_json(msgs, label="pose3d", temperature=temperature)
    return data.get("objects", [])


def _key(det: dict) -> str:
    c = (det.get("color") or "").lower()
    cat = (det.get("category") or "object").lower()
    return f"{c}_{cat}" if c else cat


# a raw vote keeps the detection AND the camera it came from (multi-view fusion)
_Vote = Tuple[Camera, dict]


def _collect(image_path: str, camera: Camera, goal: str, n_agents: int) -> List[_Vote]:
    temps = [0.1, 0.5, 0.8, 0.3, 0.6][:n_agents]
    out: List[_Vote] = []
    for temp in temps:
        for det in _query_agent(image_path, goal, temp):
            out.append((camera, det))
    return out


def _fuse(votes: List[_Vote], *, block_h: float, table_z: float) -> Dict[str, PoseEstimate]:
    # group raw votes by object key
    by_key: Dict[str, List[_Vote]] = {}
    for cam, det in votes:
        by_key.setdefault(_key(det), []).append((cam, det))

    out: Dict[str, PoseEstimate] = {}
    for key, vs in by_key.items():
        # pass 1: stack_level is occlusion-robust -> take the MAX any view reported
        # (a view that can see the stack reports >0; views that miss it report 0)
        lvl = max(int(d.get("stack_level", 0) or 0) for _, d in vs)
        plane = table_z + lvl * block_h
        # pass 2: back-project every vote's pixel to the FUSED plane, median the xy
        xs, ys = [], []
        for cam, d in vs:
            uv = d.get("base_xy")
            if not uv or len(uv) != 2:
                continue
            p = cam.backproject_to_plane(float(uv[0]) * cam.width,
                                         float(uv[1]) * cam.height, plane_z=plane)
            if p is not None:
                xs.append(p[0]); ys.append(p[1])
        if not xs:
            continue
        mx, my = float(np.median(xs)), float(np.median(ys))
        spread = float(np.median([np.hypot(x - mx, y - my) for x, y in zip(xs, ys)])) * 100
        d0 = vs[0][1]
        color = (d0.get("color") or "").lower()
        cat = (d0.get("category") or "object").lower()
        oid = f"{color}_{cat}" if color else (d0.get("id") or key)
        out[oid] = PoseEstimate(
            object_id=oid, color=d0.get("color"), category=d0.get("category", "object"),
            xyz=np.array([mx, my, table_z + lvl * block_h + block_h / 2]),
            xy_std_cm=spread, n_votes=len(xs), stack_level=lvl,
        )
    return out


def estimate_poses(image_path: str, camera: Camera, *, goal: str = "",
                   n_agents: int = 3, block_h: float = 0.04, table_z: float = 0.0
                   ) -> Dict[str, PoseEstimate]:
    """Single-view multi-agent estimate."""
    return _fuse(_collect(image_path, camera, goal, n_agents),
                 block_h=block_h, table_z=table_z)


def estimate_poses_multiview(views: List[Tuple[str, Camera]], *, goal: str = "",
                             n_agents: int = 2, block_h: float = 0.04, table_z: float = 0.0
                             ) -> Dict[str, PoseEstimate]:
    """Fuse a council across MULTIPLE views by TRIANGULATION.

    For each object, take its base pixel in each view (median over that view's
    agents), form rays, and triangulate the true 3D point. Height -> stack_level
    directly, so we don't depend on the VLM's noisy per-image stack label. Falls
    back to single-view plane back-projection for objects seen by only one view.
    """
    # collect per-(key, view) pixels
    per_key: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    cams: List[Camera] = [c for _, c in views]
    meta: Dict[str, dict] = {}
    for vi, (img, cam) in enumerate(views):
        for _, det in _collect(img, cam, goal, n_agents):
            uv = det.get("base_xy")
            if not uv or len(uv) != 2:
                continue
            k = _key(det)
            per_key.setdefault(k, {}).setdefault(vi, []).append(
                (float(uv[0]) * cam.width, float(uv[1]) * cam.height))
            meta.setdefault(k, det)

    out: Dict[str, PoseEstimate] = {}
    for k, view_px in per_key.items():
        # one median pixel per view -> one ray per view
        rays = []
        for vi, pxs in view_px.items():
            u = float(np.median([p[0] for p in pxs]))
            v = float(np.median([p[1] for p in pxs]))
            rays.append(cams[vi].ray(u, v))
        p = _triangulate_or_plane(rays, cams, view_px, table_z)
        if p is None:
            continue
        lvl = max(0, int(round((p[2] - table_z - block_h / 2) / block_h)))
        d0 = meta[k]
        color = (d0.get("color") or "").lower()
        cat = (d0.get("category") or "object").lower()
        oid = f"{color}_{cat}" if color else (d0.get("id") or k)
        out[oid] = PoseEstimate(
            object_id=oid, color=d0.get("color"), category=d0.get("category", "object"),
            xyz=np.array([p[0], p[1], p[2]]),
            xy_std_cm=0.0, n_votes=sum(len(x) for x in view_px.values()), stack_level=lvl,
        )
    return out


def _triangulate_or_plane(rays, cams, view_px, table_z):
    from .camera import triangulate
    if len(rays) >= 2:
        return triangulate(rays)
    # single view -> intersect the lone ray with the table plane
    o, d = rays[0]
    if abs(d[2]) < 1e-9:
        return None
    s = (table_z - o[2]) / d[2]
    return o + s * d if s > 0 else None
