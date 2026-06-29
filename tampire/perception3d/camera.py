"""Pinhole camera model: project world<->pixel and back-project a pixel onto a
known table plane (ray-plane intersection). This is what turns a VLM's 2D pixel
guess into a metric 3D position.

Convention matches MuJoCo / OpenGL cameras: the camera looks down its local -Z,
local +X is right, local +Y is up. `R` (cam_xmat, 3x3) maps camera-frame -> world.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Camera:
    pos: np.ndarray            # (3,) world position
    R: np.ndarray              # (3,3) camera->world rotation (columns: right, up, backward)
    fovy_deg: float            # vertical field of view
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def _tan_half_fovy(self) -> float:
        return math.tan(math.radians(self.fovy_deg) / 2.0)

    def project(self, world_xyz) -> Optional[Tuple[float, float]]:
        """World point -> pixel (u, v). None if behind the camera."""
        p = np.asarray(world_xyz, float) - self.pos
        p_cam = self.R.T @ p                     # world -> camera frame
        z = p_cam[2]
        if z >= 0:                               # camera looks along -Z; +Z is behind
            return None
        t = self._tan_half_fovy()
        x_ndc = (p_cam[0] / (-z)) / (t * self.aspect)
        y_ndc = (p_cam[1] / (-z)) / t
        u = (x_ndc + 1.0) * 0.5 * self.width
        v = (1.0 - (y_ndc + 1.0) * 0.5) * self.height   # image v grows downward
        return float(u), float(v)

    def ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel -> (origin, unit direction) ray in world frame."""
        t = self._tan_half_fovy()
        x_ndc = (2.0 * u / self.width) - 1.0
        y_ndc = 1.0 - (2.0 * v / self.height)
        dir_cam = np.array([x_ndc * t * self.aspect, y_ndc * t, -1.0])
        dir_world = self.R @ dir_cam
        dir_world /= np.linalg.norm(dir_world)
        return self.pos.copy(), dir_world

    def backproject_to_plane(self, u: float, v: float, plane_z: float = 0.0) -> Optional[np.ndarray]:
        """Pixel -> world point where its ray meets the horizontal plane z=plane_z."""
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        s = (plane_z - o[2]) / d[2]
        if s <= 0:
            return None
        return o + s * d


def triangulate(rays) -> Optional[np.ndarray]:
    """Least-squares 3D point closest to a set of rays [(origin, unit dir), ...].

    Minimises Σ ||(I - d d^T)(p - o)||² -> linear system A p = b. This recovers
    true 3D (including height) from multiple views, so object stacking falls out of
    the estimated z rather than a noisy per-image label. Needs >= 2 non-parallel rays.
    """
    rays = list(rays)
    if len(rays) < 2:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        d = np.asarray(d, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ np.asarray(o, float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def from_mujoco(model, data, camera_name: str, width: int, height: int) -> Camera:
    """Build a Camera from a live MuJoCo model/data (after mj_forward/step)."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cid < 0:
        raise ValueError(f"camera '{camera_name}' not found")
    pos = np.array(data.cam_xpos[cid])
    R = np.array(data.cam_xmat[cid]).reshape(3, 3)
    fovy = float(model.cam_fovy[cid])
    return Camera(pos=pos, R=R, fovy_deg=fovy, width=width, height=height)
