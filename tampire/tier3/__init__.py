"""Tier-3: integration with NVIDIA RoboLab (Isaac Lab benchmark, 120 tasks).

RoboLab's task suite (block stacking, bowl-in-bin, mug arrangement, ...) is exactly
TAMPire's domain. This package provides a `TAMPireClient` that implements RoboLab's
`InferenceClient` ABC: per episode it perceives the camera frame + language instruction
and runs the TAMPire pipeline to a plan, then drives a low-level skill controller that
emits the per-step action chunks RoboLab expects.

HARDWARE: RoboLab runs on NVIDIA Isaac Sim and REQUIRES an NVIDIA RTX GPU (48GB+ VRAM)
on Linux. It cannot run on macOS/Apple-Silicon. This adapter is therefore validated
*against RoboLab's real interface* (the ABC was read from the published source and is
vendored in `_base_shim.py` for off-hardware testing) and self-tested here without Isaac
(`python -m tampire.tier3.selftest`), but end-to-end evaluation must run on RTX hardware.

What is fully implemented & tested off-hardware:
  - InferenceClient ABC conformance + per-env episode/plan state
  - camera-frame extraction -> TAMPire perception+planning (real Cerebras calls)
  - skill controller producing EEF-delta action chunks toward plan waypoints

What needs the target hardware/task to finalize (documented, pluggable):
  - exact action convention of the registered task (jointpos vs EEF-delta; action_dim;
    gripper sign) -> set on TAMPireClient(action_spec=...)
  - source of object poses for the skill (RoboLab policy obs is vision-only; supply a
    pose key if the task exposes privileged state, else a vision pose estimator)
"""
