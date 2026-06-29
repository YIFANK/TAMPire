"""TAMPireClient — a RoboLab InferenceClient backed by the TAMPire planner.

Per episode (per env_id): perceive the camera frame + language instruction, run the
TAMPire pipeline to a verified plan, then drive a SkillRunner that emits the per-step
action chunks RoboLab steps with. Planning is a one-shot remote call (Cerebras); the
"server" in RoboLab's server-client model is effectively the Cerebras API, so no extra
process is needed.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image

# Real base class on Isaac/RTX hardware; vendored shim off-hardware (see _base_shim).
try:
    from robolab.eval.base_client import InferenceClient  # type: ignore
except Exception:  # pragma: no cover - exercised off-hardware
    from ._base_shim import InferenceClient

from .. import pipeline
from .skills import ActionSpec, SkillRunner

# obs -> {object_id: xyz} ; tasks differ, so this is injectable.
PoseProvider = Callable[[Any, int], Dict[str, np.ndarray]]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TAMPireClient(InferenceClient):
    def __init__(
        self,
        *,
        camera_key: str = "over_shoulder_left_camera",
        instruction_override: Optional[str] = None,
        action_spec: Optional[ActionSpec] = None,
        pose_provider: Optional[PoseProvider] = None,
        myopic_planner: bool = False,
        save_frames_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.camera_key = camera_key
        self.instruction_override = instruction_override
        self.spec = action_spec or ActionSpec()
        self.pose_provider = pose_provider
        self.myopic_planner = myopic_planner
        self.save_frames_dir = save_frames_dir
        self._runners: Dict[int, SkillRunner] = {}
        self._plans: Dict[int, Any] = {}

    # ---- episode state ----------------------------------------------------
    def reset(self, *, env_id: Optional[int] = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._runners.clear()
            self._plans.clear()
        else:
            self._runners.pop(env_id, None)
            self._plans.pop(env_id, None)

    # ---- obs helpers ------------------------------------------------------
    def _camera_frame(self, raw_obs: Any, env_id: int) -> np.ndarray:
        img = _to_numpy(raw_obs["image_obs"][self.camera_key][env_id])
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img

    def _eef_pos(self, raw_obs: Any, env_id: int) -> np.ndarray:
        prop = raw_obs.get("proprio_obs", {})
        for k in ("eef_pos", "ee_pos", "ee_position", "tcp_pos"):
            if k in prop:
                return _to_numpy(prop[k][env_id]).reshape(-1)[:3]
        raise KeyError(
            "No end-effector position in proprio_obs. Set ActionSpec/obs to expose one "
            "(keys tried: eef_pos/ee_pos/ee_position/tcp_pos)."
        )

    def _object_poses(self, raw_obs: Any, env_id: int) -> Dict[str, np.ndarray]:
        if self.pose_provider is not None:
            return self.pose_provider(raw_obs, env_id)
        # RoboLab's policy obs is vision-only; without a pose source the skill cannot
        # reach metric targets. Tasks exposing privileged poses should pass pose_provider.
        raise NotImplementedError(
            "TAMPireClient needs object poses to execute. Pass pose_provider=... that maps "
            "obs -> {object_id: xyz}, or run with a task that exposes privileged object state."
        )

    # ---- planning ---------------------------------------------------------
    def _make_runner(self, raw_obs: Any, instruction: str, env_id: int) -> SkillRunner:
        frame = self._camera_frame(raw_obs, env_id)
        d = self.save_frames_dir or tempfile.gettempdir()
        os.makedirs(d, exist_ok=True)
        img_path = os.path.join(d, f"tampire_env{env_id}.png")
        Image.fromarray(frame).save(img_path)

        res = pipeline.run(instruction, image_path=img_path,
                           myopic_planner=self.myopic_planner)
        self._plans[env_id] = res.final_plan
        poses = self._object_poses(raw_obs, env_id)
        return SkillRunner.from_plan(res.final_plan, poses, self.spec)

    # ---- main control loop (override; non-chunk flow per ABC guidance) -----
    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        instr = self.instruction_override or instruction
        if env_id not in self._runners:
            self._runners[env_id] = self._make_runner(obs, instr, env_id)
        action = self._runners[env_id].step(self._eef_pos(obs, env_id))
        return {"action": action, "viz": None}

    # ---- abstract hooks: unused (infer is overridden) but ABC needs them ---
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict:
        return {"camera": self._camera_frame(raw_obs, env_id)}

    def _pack_request(self, extracted_obs: dict, instruction: str):
        raise NotImplementedError("TAMPireClient plans in infer(); no wire request.")

    def _query_server(self, request):
        raise NotImplementedError("TAMPireClient plans in infer(); no separate server.")

    def _unpack_response(self, response) -> np.ndarray:
        raise NotImplementedError("TAMPireClient plans in infer(); no server response.")

    # introspection for logging
    def plan_for(self, env_id: int = 0):
        return self._plans.get(env_id)
