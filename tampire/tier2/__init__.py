"""Tier-2: integration with a recognized robot benchmark (robosuite).

robosuite gives a real Panda arm, real agentview pixels, and a native success
criterion. TAMPire perceives the rendered pixels and plans; the plan is executed
by a scripted OSC pick-place skill (low-level control is not what TAMPire solves).

Requires `robosuite` + `mujoco==3.3.0` (newer mujoco breaks robosuite 1.5's
mj_fullM call). Install: `.venv312/bin/pip install -r requirements-tier2.txt`.

Note: PyBullet-based benchmarks (VIMA-Bench, Ravens/CLIPort) can't build on this
macOS/Python, so robosuite (MuJoCo-based) is the chosen recognized benchmark.
"""
