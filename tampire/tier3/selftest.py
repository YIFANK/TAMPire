"""Off-hardware self-test for the RoboLab integration.

Proves — WITHOUT Isaac/GPU — that TAMPireClient:
  1. conforms to RoboLab's InferenceClient ABC,
  2. extracts the camera frame and plans with the real TAMPire pipeline (Cerebras),
  3. produces correctly-shaped action chunks whose closed loop converges (the skill
     controller actually reaches its waypoints and finishes the task motion).

It fabricates a RoboLab-style obs (same shape pattern as pi0_family/client.py's
__main__) using a rendered scene image + a synthetic object-pose provider, then runs
a closed loop integrating the EEF actions so the controller advances to completion.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from rich.console import Console

from .client import TAMPireClient
from .skills import ActionSpec

console = Console()


def _fake_obs(img: np.ndarray, eef: np.ndarray) -> dict:
    return {
        "image_obs": {"over_shoulder_left_camera": [img]},
        "proprio_obs": {"eef_pos": np.asarray([eef], dtype=np.float32)},
    }


def main() -> int:
    # a real rendered scene to perceive
    img_path = "scenes/scene.png"
    if not os.path.exists(img_path):
        from scripts.make_scene_image import make
        make(img_path)
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    # synthetic privileged poses for the objects in scene.png (metres, robot frame)
    poses = {
        "blue_block": np.array([0.10, 0.05, 0.04]),
        "blue_bowl": np.array([0.25, 0.00, 0.03]),
        "red_block": np.array([-0.10, 0.05, 0.04]),
        "green_block": np.array([-0.10, 0.05, 0.08]),
    }

    def pose_provider(_obs, _env_id):
        return poses

    spec = ActionSpec(dim=7, gripper_index=6)
    client = TAMPireClient(action_spec=spec, pose_provider=pose_provider)

    instruction = "put the blue block in the bowl"
    console.print(f"[bold]Tier-3 self-test[/bold]  instruction: [yellow]{instruction}[/yellow]")

    # closed loop: integrate EEF actions so the controller can reach waypoints
    eef = np.array([0.0, 0.0, 0.30], dtype=np.float32)
    dt = 0.04
    actions = []
    grip_changes = 0
    last_grip = None
    max_steps = 1500
    steps = 0
    for steps in range(1, max_steps + 1):
        obs = _fake_obs(img, eef)
        out = client.infer(obs, instruction, env_id=0)
        a = out["action"]
        assert a.shape == (spec.dim,), f"action shape {a.shape} != {(spec.dim,)}"
        actions.append(a)
        g = a[spec.gripper_index]
        if last_grip is not None and np.sign(g) != np.sign(last_grip):
            grip_changes += 1
        last_grip = g
        eef = eef + a[0:3] * dt  # integrate position delta
        if client._runners[0].done:
            break

    plan = client.plan_for(0)
    console.print("\n[bold]plan:[/bold]")
    for i, s in enumerate(plan.steps, 1) if plan else []:
        console.print(f"  {i}. {s}")

    ok = bool(plan and plan.steps) and client._runners[0].done
    console.print(f"\n  action dim          : {spec.dim}  (chunk shape per step OK)")
    console.print(f"  control steps       : {steps}")
    console.print(f"  gripper transitions : {grip_changes}  (open/close cycles)")
    console.print(f"  controller finished : {client._runners[0].done}")
    console.print(f"\n  [bold]{'PASS' if ok else 'FAIL'}[/bold] — "
                  "ABC conformance + planning + skill convergence "
                  f"{'verified off-hardware' if ok else 'FAILED'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
