"""Launcher to evaluate TAMPire as a policy inside RoboLab. Mirrors the structure
of RoboLab's `policies/pi0_family/run.py`.

RUN ON RTX/Isaac HARDWARE ONLY. To use: copy this repo's `tampire/` onto a machine
with RoboLab installed (NVIDIA RTX GPU + Linux + Isaac Sim), then:

    python -m tampire.tier3.run --task StackYellowOnRedTask --num-envs 10

The Isaac imports are deferred into main() so this module stays importable (and the
client stays unit-testable) on machines without Isaac.
"""
from __future__ import annotations

import argparse


def make_client(args):
    from .client import TAMPireClient
    from .skills import ActionSpec
    return TAMPireClient(
        camera_key=args.camera_key,
        myopic_planner=args.baseline,
        action_spec=ActionSpec(dim=args.action_dim, gripper_index=args.action_dim - 1),
        # NOTE: supply a pose_provider matching your task; see README. RoboLab policy
        # obs is vision-only, so privileged poses (or a vision estimator) are required.
        pose_provider=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TAMPire in RoboLab.")
    parser.add_argument("--camera-key", default="over_shoulder_left_camera")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--baseline", action="store_true",
                        help="myopic planner (exercises the repair council)")

    # ---- Isaac is required from here on ----
    import cv2  # noqa: F401  -- must import before isaaclab (RoboLab requirement)
    from isaaclab.app import AppLauncher

    from robolab.eval.runner import add_common_eval_args, run_evaluation
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import robolab.constants  # noqa: F401,E402

    return run_evaluation(args_cli, policy="tampire", client_factory=make_client)


if __name__ == "__main__":
    raise SystemExit(main())
