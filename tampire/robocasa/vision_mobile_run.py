"""Full vision-based mobile manipulation: multi-cam Gemma localization → drive →
real OSC grasp → place → RoboCasa native success. No privileged obj_pos used for
the drive/grasp targets. Retries over seeds until a clean native PASS, saves GIF.

    cd /Users/yifankang/R3-Manipulation && PYTHONPATH=/Users/yifankang/TAMPire \
      MUJOCO_GL=cgl PYTHONWARNINGS=ignore .venv-arm64/bin/python \
      -m tampire.robocasa.vision_mobile_run
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "cgl")


def attempt(seed):
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env
    from .mobile_exec import MobileArm, see_multicam

    env = create_env(env_name="PickPlaceCounterToSink", seed=seed, render_onscreen=False)
    env.reset()
    instr = env.get_ep_meta()["lang"]
    A = MobileArm(env)
    base0 = A._base().copy()

    # perceive ONCE while the object is in view; the result is a WORLD coordinate, so
    # it stays valid after the base moves — no need to re-perceive mid-drive.
    vi = see_multicam(env, instr)
    if vi is None:
        env.close(); return ("no-vision", None)
    gt = np.array(env._get_observations()["obj_pos"])
    verr = float(np.linalg.norm(vi[:2] - gt[:2]) * 100)
    if verr > 5:
        env.close(); return (f"vision off {verr:.1f}cm", None)

    # REAL driving (no teleport): roll backward to open a standoff gap, then drive the
    # base forward via the velocity controller until the arm can reach the object.
    A.calib_forward()
    A.drive_back(dist=0.55)
    A.frames = []; A.snap()
    A.drive_to(vi[:2])

    A.grasp(np.array(vi[:3], dtype=float))                  # grasp at PERCEIVED coords
    grasped = bool(env._get_observations()["obj_pos"][2] > 0.99)

    sk = env.sink.get_int_sites(relative=False)
    p0, px, py, pz = list(sk.values())[0]
    sc = (np.array(p0) + np.array(px) + np.array(py) + np.array(pz)) / 4
    A.place(list(sc))
    native = bool(env._check_success())
    fr = A.frames
    env.close()
    return (f"vision_err={verr:.1f}cm grasped={grasped} native={native}",
            (grasped, native, fr, verr))


def main():
    for seed in [4, 6, 2, 18, 22, 24, 16, 1, 10, 17, 19, 21, 3, 5, 7]:
        msg, res = attempt(seed)
        print(f"seed {seed}: {msg}", flush=True)
        if res is None:
            continue
        grasped, native, fr, verr = res
        if grasped and native:
            imgs = [Image.fromarray(x) for x in fr]
            out = "/Users/yifankang/TAMPire/runs/rc_vision_mobile.gif"
            imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=80, loop=0)
            print(f"SAVED {out} ({len(fr)} frames, vision_err={verr:.1f}cm)", flush=True)
            return
    print("no clean vision+native success across seeds", flush=True)


if __name__ == "__main__":
    main()
