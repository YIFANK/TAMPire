"""Two-branch perception (TiPToP-style), GPU-free.

  semantic branch : Gemma-4 council -> which objects exist (colour, block/bowl) +
                    goal grounding. Runs on ONE view (cheap).
  3D branch       : multi-view depth -> per-object point clouds -> centroids,
                    z-extents (stacking), convex-hull meshes (collision).

Fused into an object-centric scene representation: meshes + symbolic predicates,
exactly the input TAMP wants. No privileged state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import camera as C
from . import depth as D
from . import estimator as E

_VIEWS = ("angled", "angled_left", "angled_right")


@dataclass
class PerceivedScene:
    scene: object                       # tampire.schemas.Scene
    geoms: Dict[str, D.ObjectGeom]      # color -> geometry (+ hull)
    categories: Dict[str, str]          # color -> "block" | "bowl"

    def hulls(self) -> Dict[str, np.ndarray]:
        out = {}
        for c, g in self.geoms.items():
            if g.hull_vertices is not None:
                cat = self.categories.get(c, "block")
                out[f"{c}_{'bowl' if cat in ('bowl','cup','plate','tray') else 'block'}"] = g.hull_vertices
        return out


def perceive(env, goal: str, *, views=_VIEWS, n_agents: int = 2,
             table_z: float = 0.0) -> PerceivedScene:
    cams: List[Tuple[str, C.Camera]] = [
        (n, C.from_mujoco(env.model, env.data, n, 640, 480)) for n in views
    ]

    # --- semantic branch: council tells us which colours are blocks vs bowls ---
    sem = E.estimate_poses_multiview([(_render(env, n), cam) for n, cam in cams],
                                     goal=goal, n_agents=n_agents)
    categories: Dict[str, str] = {}
    for pe in sem.values():
        col = D.normalize_color(pe.color or "")  # map synonyms onto our palette
        if col:
            categories[col] = pe.category or "block"

    # --- 3D branch: localize the council-detected colours via depth. Gating on the
    # council's (normalized) colour set avoids depth hallucinating objects from stray
    # pixels, while depth — not the council — owns metric position/stacking. ---
    colors = list(categories) or [c for c in (D.normalize_color(pe.color or "")
                                              for pe in sem.values()) if c]
    geoms = D.per_object_geom(env, cams, colors, table_z=table_z)
    scene = D.scene_from_depth(geoms, categories, table_bounds=env.scene.table_bounds)
    return PerceivedScene(scene=scene, geoms=geoms, categories=categories)


def _render(env, camera_name: str) -> str:
    import os
    import tempfile

    from PIL import Image
    d = tempfile.mkdtemp(prefix="tampire_sem_")
    p = os.path.join(d, f"{camera_name}.png")
    Image.fromarray(env.render(camera_name)).save(p)
    return p
