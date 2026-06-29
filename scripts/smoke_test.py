"""Verify the Cerebras endpoint: model list, a timed text call, and a vision call.
Run this first at a new venue / on a new key."""
import sys
import time

sys.path.insert(0, ".")

from tampire import llm
from tampire.config import CONFIG


def main() -> int:
    CONFIG.require_key()
    print(f"model = {CONFIG.model}   base = {CONFIG.base_url}")

    # text
    t0 = time.time()
    text, m = llm.chat([llm.user("Reply with exactly: PONG")], label="smoke-text", max_tokens=10)
    print(f"[text]  reply={text!r}  wall={time.time()-t0:.3f}s  model_compute={m.model_s*1000:.1f}ms")

    # vision (generate a tiny scene, then ask about it)
    from scripts.make_scene_image import make
    img = "scenes/scene.png"
    make(img)
    text, m = llm.chat(
        [llm.user_with_image("List the colored objects you see, terse.", img)],
        label="smoke-vision", max_tokens=60,
    )
    print(f"[vision] image_tokens={m.image_tokens}  reply={text.strip()!r}")
    if m.image_tokens > 0:
        print("OK — endpoint is multimodal. 'From pixels' is live. ✅")
    else:
        print("WARNING — no image_tokens reported; endpoint may not be multimodal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
