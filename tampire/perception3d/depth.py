"""Depth-based 3D perception — the TiPToP-style geometry branch, GPU-free.

Instead of a stereo network (FoundationStereo) we use the simulator's depth render;
instead of SAM we color-segment the RGB (our objects are distinctly coloured). Per
object we fuse a point cloud across multiple camera views, take a robust centroid for
the pose, read stack relations straight off the z-extents, and build a convex-hull
mesh for collision checking (over-approximate, "surprisingly sufficient" per TiPToP).

The Gemma council still owns the *semantic* branch (which objects exist, goal grounding);
this module owns metric geometry. No privileged object state is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera import Camera

# reference block colours (0-255), matching simreal/mjscene _RGBA
COLOR_RGB = {
    "red": (217, 46, 46), "green": (43, 158, 74), "blue": (48, 102, 214),
    "yellow": (242, 194, 13), "orange": (232, 115, 26), "purple": (143, 69, 173),
    "cyan": (26, 181, 196),
}


@dataclass
class ObjectGeom:
    object_id: str
    color: str
    points: np.ndarray            # (N,3) fused world points
    centroid: np.ndarray          # (3,) robust centre (median)
    z_min: float
    z_max: float
    hull_vertices: Optional[np.ndarray] = None   # (V,3) convex hull verts (collision mesh)

    @property
    def xy(self) -> np.ndarray:
        return self.centroid[:2]


# ---------------------------------------------------------------------------
# rendering + unprojection
# ---------------------------------------------------------------------------
def render_rgbd(env, camera: str, w: int = 640, h: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 HxWx3, depth float HxW) for a TabletopEnv-like object."""
    import mujoco
    rgb = env.render(camera) if hasattr(env, "render") else None
    r = mujoco.Renderer(env.model, h, w)
    try:
        r.enable_depth_rendering()
        r.update_scene(env.data, camera=camera)
        depth = r.render().copy()
    finally:
        r.close()
    if rgb is None or rgb.shape[:2] != (h, w):
        rr = mujoco.Renderer(env.model, h, w)
        rr.update_scene(env.data, camera=camera)
        rgb = rr.render().copy()
        rr.close()
    return rgb, depth


def unproject(depth: np.ndarray, cam: Camera) -> np.ndarray:
    """Perpendicular-depth unprojection -> (H*W, 3) world points."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    th = np.tan(np.radians(cam.fovy_deg) / 2.0)
    x_ndc = (2 * (us + 0.5) / W - 1) * th * cam.aspect
    y_ndc = (1 - 2 * (vs + 0.5) / H) * th
    pc = np.stack([x_ndc * depth, y_ndc * depth, -depth], axis=-1).reshape(-1, 3)
    return (cam.R @ pc.T).T + cam.pos


# common VLM colour synonyms -> our palette
COLOR_SYNONYM = {
    "navy": "blue", "dark_blue": "blue", "darkblue": "blue", "royal_blue": "blue",
    "light_blue": "cyan", "teal": "cyan", "turquoise": "cyan", "lime": "green",
    "magenta": "purple", "violet": "purple", "gold": "yellow", "amber": "orange",
}


def normalize_color(color: str) -> Optional[str]:
    c = (color or "").lower().strip().replace(" ", "_")
    if c in COLOR_RGB:
        return c
    return COLOR_SYNONYM.get(c)


def color_mask(rgb: np.ndarray, color: str, tol: float = 55.0) -> np.ndarray:
    """Boolean mask of pixels matching a reference block colour (nearest-colour)."""
    color = normalize_color(color)
    if color is None:
        return np.zeros(rgb.shape[:2], bool)
    ref = np.array(COLOR_RGB[color], float)
    d = np.linalg.norm(rgb.astype(float) - ref, axis=-1)
    # also require this be the NEAREST known colour (avoids table/shadow false hits)
    dists = {c: np.linalg.norm(rgb.astype(float) - np.array(v, float), axis=-1)
             for c, v in COLOR_RGB.items()}
    nearest = np.argmin(np.stack(list(dists.values())), axis=0)
    this_idx = list(COLOR_RGB).index(color)
    return (d < tol) & (nearest == this_idx)


# ---------------------------------------------------------------------------
# per-object geometry across views
# ---------------------------------------------------------------------------
def per_object_geom(env, views: List[Tuple[str, Camera]], colors: List[str],
                    *, table_z: float = 0.0, min_pts: int = 40) -> Dict[str, ObjectGeom]:
    """Fuse per-colour point clouds across views and build ObjectGeoms."""
    from scipy.spatial import ConvexHull

    acc: Dict[str, List[np.ndarray]] = {c: [] for c in colors}
    for cam_name, cam in views:
        rgb, depth = render_rgbd(env, cam_name, cam.width, cam.height)
        world = unproject(depth, cam)
        valid = (depth.reshape(-1) > 0) & (depth.reshape(-1) < 5)
        above = world[:, 2] > table_z + 0.004           # drop the table plane
        for c in colors:
            m = color_mask(rgb, c).reshape(-1) & valid & above
            if m.any():
                acc[c].append(world[m])

    out: Dict[str, ObjectGeom] = {}
    for c, chunks in acc.items():
        if not chunks:
            continue
        pts = np.concatenate(chunks, axis=0)
        if len(pts) < min_pts:
            continue
        z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
        # robust median centroid. (Visible-surface sampling biases xy toward the
        # camera-facing faces by up to ~half the object width; using symmetric views
        # keeps this small and consistent — ~2.5 cm here — and outlier-free.)
        centroid = np.array([np.median(pts[:, 0]), np.median(pts[:, 1]),
                             (z_min + z_max) / 2.0])
        hull_v = None
        try:
            if len(pts) >= 8:
                hull_v = pts[ConvexHull(pts).vertices]
        except Exception:
            hull_v = None
        out[c] = ObjectGeom(
            object_id=c, color=c, points=pts, centroid=centroid,
            z_min=z_min, z_max=z_max, hull_vertices=hull_v)
    return out


def scene_from_depth(geoms: Dict[str, "ObjectGeom"], categories: Dict[str, str],
                     table_bounds=(-0.4, -0.3, 0.4, 0.3), bowl_radius: float = 0.07):
    """Build a TAMPire Scene from depth geoms + the council's per-colour categories.
    Stacking comes from z-extents (reliable), containment from bowl footprints."""
    from ..schemas import Predicate, Scene, WorldObject

    objs, preds = [], []
    cat = {c: categories.get(c, "block") for c in geoms}
    for c, g in geoms.items():
        is_bowl = cat[c] in ("bowl", "cup", "plate", "tray")
        objs.append(WorldObject(
            id=f"{c}_{'bowl' if is_bowl else 'block'}", category=("bowl" if is_bowl else "block"),
            color=c, position=tuple(float(x) for x in g.centroid),
            size=((0.12, 0.12, 0.05) if is_bowl else (0.04, 0.04, 0.04)),
            affordances=(["container", "support"] if is_bowl else ["graspable", "stackable"])))

    def oid(c):
        return f"{c}_{'bowl' if cat[c] in ('bowl','cup','plate','tray') else 'block'}"

    blocks = [c for c in geoms if cat[c] not in ("bowl", "cup", "plate", "tray")]
    bowls = [c for c in geoms if cat[c] in ("bowl", "cup", "plate", "tray")]
    support = {}
    for c in blocks:
        g = geoms[c]
        below = [b for b in blocks if b != c and geoms[b].z_max < g.z_max - 0.005
                 and np.linalg.norm(g.xy - geoms[b].xy) < 0.045
                 and abs(g.z_min - geoms[b].z_max) < 0.03]
        below.sort(key=lambda b: g.z_min - geoms[b].z_max)
        in_bowl = next((w for w in bowls if np.linalg.norm(g.xy - geoms[w].xy) < bowl_radius), None)
        if below:
            support[oid(c)] = oid(below[0]); preds.append(Predicate("on", [oid(c), oid(below[0])]))
        elif in_bowl is not None:
            support[oid(c)] = oid(in_bowl); preds.append(Predicate("in", [oid(c), oid(in_bowl)]))
        else:
            support[oid(c)] = "table"; preds.append(Predicate("on", [oid(c), "table"]))
    used = set(support.values())
    for o in objs:
        if o.id not in used:
            preds.append(Predicate("clear", [o.id]))
    for c in blocks:
        preds.append(Predicate("graspable", [oid(c)]))
    for c in bowls:
        preds.append(Predicate("container", [oid(c)]))
    return Scene(objects=objs, predicates=preds, table_bounds=table_bounds,
                 notes="grounded from depth point clouds (TiPToP-style 3D branch)")
