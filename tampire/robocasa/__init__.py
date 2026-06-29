"""RoboCasa adapter (Milestone A): perceive a real kitchen task's image + natural-
language instruction, run the TAMPire multi-agent pipeline, and emit a *verified*
long-horizon TAMP plan.

RoboCasa is robosuite+MuJoCo (no Isaac Sim), so it runs on this Mac. It must run in
the RoboCasa venv (mujoco==3.3.1), e.g.:

    cd /Users/yifankang/R3-Manipulation
    PYTHONPATH=/Users/yifankang/TAMPire MUJOCO_GL=cgl \\
      .venv-arm64/bin/python -m tampire.robocasa --task PickPlaceCounterToCabinet --seed 3

The robocasa import is deferred into task.py so importing tampire elsewhere never
requires robocasa to be installed.
"""
