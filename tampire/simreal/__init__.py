"""Tier-1 real simulator (MuJoCo).

Real rigid-body physics + real rendered pixels + an INDEPENDENT success check
read from the physics state. Manipulation is abstracted ("magic" grasp, like
Ravens' suction oracle) so we don't need a full arm + IK controller — the point
is to ground perception in real pixels and verdicts in real physics, not to
solve low-level control.

Requires `mujoco` (install in a Python 3.11/3.12 venv; cp39 has no wheels here).
"""
