# Tier-3: TAMPire × NVIDIA RoboLab

RoboLab ([NVlabs/RoboLab](https://github.com/NVlabs/RoboLab)) is an Isaac-Lab benchmark
with **120 manipulation tasks** (block stacking, bowl-in-bin, mug arrangement, …) and
native success detection — exactly TAMPire's domain. This package plugs TAMPire in as a
RoboLab **policy** via its `InferenceClient` interface.

## ⚠️ Hardware
RoboLab runs on **NVIDIA Isaac Sim** and requires an **NVIDIA RTX GPU (48 GB+ VRAM) on
Linux**. It does **not** run on macOS / Apple Silicon. This adapter was therefore:
- **validated against RoboLab's real interface** — the `InferenceClient` ABC was read from
  the published source and is vendored in [`_base_shim.py`](_base_shim.py) (Apache-2.0,
  © NVIDIA) for off-hardware import;
- **self-tested end-to-end without Isaac** via `python -m tampire.tier3.selftest`, which
  drives the client with a synthetic RoboLab-style observation and confirms ABC
  conformance, real TAMPire planning (Cerebras), and skill-controller convergence;
- **not executed inside RoboLab here**, because the GPU/OS aren't available.

## How it maps
`TAMPireClient(InferenceClient)` overrides `infer()` (the ABC explicitly supports this for
non-chunk policies). Per episode, per env:
1. extract the camera frame from `obs["image_obs"][camera_key]`;
2. run the TAMPire pipeline `instruction + frame → perception → grounding → plan →
   debate-repair → verified plan` (one Cerebras round-trip — the "policy server" in
   RoboLab's server-client model is effectively the Cerebras API);
3. expand the plan into EEF waypoints and emit per-step actions via `SkillRunner`.

## Two things to set for your target task
RoboLab's policy obs is **vision-only** and tasks differ in action convention, so:
- **`action_spec`** — match the registered task's controller. Default is 7-D
  differential-IK pose `[dx,dy,dz, drx,dry,drz, gripper]`. Joint-position tasks (e.g. the
  Pi0 DROID client) need an IK layer wrapping `SkillRunner`.
- **`pose_provider`** — `obs -> {object_id: xyz}`. TAMPire ships a vision-only estimator
  for this (`tampire/perception3d/`): a multi-agent council localizes object bases across
  several RoboLab cameras and **triangulates** metric 3D — no privileged state. Build a
  `Camera` per RoboLab camera (intrinsics/extrinsics from the Isaac/MuJoCo model, as in
  `perception3d.camera.from_mujoco`), then call `estimate_poses_multiview`. This is the
  same path validated in Tier-1 (~1.6 cm) and used to grasp on the real Panda in Tier-2
  (`tampire.tier2 --vision`).

## Run on RTX hardware
```bash
# on a machine with RoboLab installed:
pip install -e .          # or copy tampire/ next to robolab/
python -m tampire.tier3.run --task StackYellowOnRedTask --num-envs 10
```
See [`run.py`](run.py) — it mirrors `policies/pi0_family/run.py`, deferring Isaac imports
into `main()` so the client stays importable/testable off-hardware.
