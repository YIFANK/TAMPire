"""Speed in Action: Cerebras vs a GPU provider on the SAME multi-agent workload.

Replays a representative TAMPire reasoning workload (planner + 3 critics + repair —
the calls a debate round actually makes) against two OpenAI-compatible backends and
reports per-call latency and sustained throughput.

FAIR COMPARISON: the SAME model (gemma-4-31b-it) runs on both backends — Cerebras
wafer-scale vs Together's GPU serving — on identical prompts with the same token budget.
This isolates the serving stack's inference speed, not model size or quality.

    cd /Users/yifankang/TAMPire && .venv-arm64-or-any/bin/python -m tampire.eval.speed_compare
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_env():
    for line in open(os.path.join(ROOT, ".env")):
        m = re.match(r'\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.*)\s*', line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


# representative debate-round workload (text reasoning, the council's calls)
WORKLOAD = [
    "You are a robot task planner. Give an ordered plan to put a mug from the counter into a "
    "closed cabinet. List the steps with one-line rationales.",
    "Critic (preconditions lens): the plan tries to place the mug into the cabinet before "
    "opening it. In 3 sentences, diagnose the precondition failure and the minimal fix.",
    "Critic (geometry lens): the cabinet is 1.2m from the robot base, beyond arm reach. In 3 "
    "sentences, diagnose the reachability failure and the minimal fix.",
    "Critic (goal lens): does the plan actually satisfy in(mug, cabinet)? In 3 sentences, "
    "explain what is missing.",
    "Repair chair: given the three critiques (open the cabinet first; drive the base within "
    "reach; ensure the mug ends inside), output the corrected ordered plan.",
]

BACKENDS = [
    {"name": "Cerebras · gemma-4-31b", "url": "https://api.cerebras.ai/v1/chat/completions",
     "key": "CEREBRAS_API_KEY", "model": "gemma-4-31b"},
    {"name": "Together (GPU) · gemma-4-31b-it",
     "url": "https://api.together.xyz/v1/chat/completions",
     "key": "TOGETHER_API_KEY", "model": "google/gemma-4-31b-it"},
]


def _call(be, prompt, max_tokens=350):
    body = {"model": be["model"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    t0 = time.time()
    r = requests.post(be["url"], json=body,
                      headers={"Authorization": f"Bearer {os.environ[be['key']]}"}, timeout=90)
    dt = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:120]}")
    ct = r.json().get("usage", {}).get("completion_tokens", 0)
    return dt, ct


def run():
    _load_env()
    results = {}
    for be in BACKENDS:
        print(f"\n=== {be['name']} ===")
        times, toks = [], []
        try:
            for i, p in enumerate(WORKLOAD):
                dt, ct = _call(be, p)
                times.append(dt); toks.append(ct)
                print(f"  call {i+1}: {dt:5.2f}s  {ct:4d} tok  {ct/dt:6.0f} tok/s")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        total_t, total_tok = sum(times), sum(toks)
        results[be["name"]] = {
            "total_s": round(total_t, 2), "total_tokens": total_tok,
            "tok_per_s": round(total_tok / total_t, 1),
            "per_call_s": [round(t, 2) for t in times],
        }
        print(f"  → debate workload: {total_t:.2f}s total, {total_tok} tok, "
              f"{total_tok/total_t:.0f} tok/s sustained")

    names = list(results)
    if len(names) == 2:
        a, b = results[names[0]], results[names[1]]
        speedup = b["total_s"] / a["total_s"] if a["total_s"] else 0
        print(f"\n[SUMMARY] {names[0]} ran the full debate workload "
              f"{speedup:.1f}× faster than {names[1]} "
              f"({a['total_s']}s vs {b['total_s']}s; "
              f"{a['tok_per_s']} vs {b['tok_per_s']} tok/s).")
    out = os.path.join(ROOT, "runs", "speed_compare.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    run()
