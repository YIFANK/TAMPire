"""Vision-based 3D pose estimation — the multi-agent grounding that replaces
privileged object poses.

A council of Gemma-4 vision agents localizes each object's base in the image; the
pixels are back-projected to metric table coordinates through the known camera
geometry, then fused (robust median + spread) across agents. The resulting poses
feed TAMP: the symbolic plan, the geometric reach/collision feasibility check, and
the motion waypoints are all computed from *perceived* geometry — no privileged state.
"""
