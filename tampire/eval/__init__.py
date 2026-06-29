"""Tier-0 evaluation: procedurally generated scenes + an ablation harness.

The symbolic verifier (sim/feasibility.py) is plain deterministic code, NOT the
LLM — so a plan passing it is a genuine check on the LLM's output, not the LLM
grading itself. We also inject ground-truth goal predicates (we generated the
goal) so the goalspec step can't confound the success metric.
"""
