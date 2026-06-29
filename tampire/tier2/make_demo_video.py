"""Stitch the full 60-second TAMPire demo video from the rendered clips + generated
title/section/speed cards. Output: runs/TAMPire_demo_60s.mp4 (+ .gif).

    python -m tampire.tier2.make_demo_video
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

R = "/Users/yifankang/TAMPire/runs"
W, H, FPS = 1180, 600, 15
BG = (8, 11, 17); TXT = (223, 231, 241); DIM = (124, 138, 160); ACC = (78, 161, 255)
GOOD = (55, 211, 154); BAD = (255, 122, 134); WARN = (255, 200, 97)


_LATO = os.path.expanduser("~/Library/Fonts")
_FONTS = {"black": _LATO + "/Lato-Black.ttf", "bold": _LATO + "/Lato-Bold.ttf",
          "regular": _LATO + "/Lato-Regular.ttf", "light": _LATO + "/Lato-Light.ttf",
          "mono": "/System/Library/Fonts/SFNSMono.ttf"}


def F(sz, weight="regular", mono=False):
    path = _FONTS["mono"] if mono else _FONTS.get(weight, _FONTS["regular"])
    try:
        return ImageFont.truetype(path, sz)
    except Exception:
        return ImageFont.load_default()


def _tracked(d, x, y, s, font, fill, track=0):
    """Draw text with optional letter-spacing (tracking) in px."""
    if not track:
        d.text((x, y), s, font=font, fill=fill); return d.textlength(s, font=font)
    cx = x
    for ch in s:
        d.text((cx, y), ch, font=font, fill=fill)
        cx += d.textlength(ch, font=font) + track
    return cx - x


def _tracklen(d, s, font, track):
    return sum(d.textlength(ch, font=font) + track for ch in s) - track if s else 0


def ctext(d, cx, y, s, font, fill, track=0):
    w = _tracklen(d, s, font, track) if track else d.textlength(s, font=font)
    _tracked(d, cx - w / 2, y, s, font, fill, track)


def load_gif(path, every=1):
    g = Image.open(path)
    fr = [f.convert("RGB") for f in ImageSequence.Iterator(g)]
    return fr[::every]


def load_video(path, seconds, fps=FPS, start=0.0):
    """Load a screen recording, trim to [start, start+seconds], resample to `fps`,
    letterbox each frame onto the demo canvas. Robust to lying container metadata
    (some screen recordings report half their true fps) by counting frames first."""
    import imageio.v2 as imageio
    meta = imageio.get_reader(path).get_meta_data()
    dur = meta.get("duration") or 0
    # true frame count by iteration — count_frames()/meta fps are unreliable here
    n_true = sum(1 for _ in imageio.get_reader(path))
    true_fps = (n_true / dur) if dur > 0 else meta.get("fps", 30)
    lo = int(start * true_fps)
    hi = min(int((start + seconds) * true_fps), n_true - 1)
    n_out = max(1, int(seconds * fps))
    targets = {int(lo + (hi - lo) * k / max(1, n_out - 1)) for k in range(n_out)}
    out = []
    for i, fr in enumerate(imageio.get_reader(path)):
        if i in targets:
            out.append(fit(Image.fromarray(fr)))
        if i > hi:
            break
    return out


def fit(im, w=W, h=H):
    """Letterbox an image onto a w×h dark canvas."""
    c = Image.new("RGB", (w, h), BG)
    s = min(w / im.width, h / im.height)
    im2 = im.resize((int(im.width * s), int(im.height * s)))
    c.paste(im2, ((w - im2.width) // 2, (h - im2.height) // 2))
    return c


def hold(img, n):
    return [img.copy() for _ in range(n)]


def load_still(path, seconds):
    """Letterbox a static image and hold it for `seconds`."""
    return hold(fit(Image.open(path).convert("RGB")), int(seconds * FPS))


# ---- cards ----
def _logo(d, cx, y, sz):
    big = F(sz, "black")
    full = _tracklen(d, "TAMPire", big, 1)
    x = cx - full / 2
    x += _tracked(d, x, y, "TAMP", big, TXT, 1)
    _tracked(d, x, y, "ire", big, ACC, 1)


def title_card(n):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    _logo(d, W / 2, 150, 76)
    ctext(d, W / 2, 272, "A Multi-Agent Zero-Shot Robotics Planner from Pixels", F(27, "light"), TXT)
    ctext(d, W / 2, 322, "Gemma-4-31B   ×   Cerebras", F(19, "bold"), DIM, track=1)
    ctext(d, W / 2, 404, "PERCEPTION · MULTI-AGENT PLANNING · CLOSED-LOOP RECOVERY · REAL ROBOT EXECUTION",
          F(13, "bold"), (96, 110, 132), track=2)
    return hold(img, n)


def section_card(num, title, sub, n):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.rounded_rectangle((W / 2 - 32, 244, W / 2 + 32, 308), 14, fill=(13, 30, 45), outline=ACC, width=2)
    ctext(d, W / 2, 250, num, F(38, "black"), ACC)
    ctext(d, W / 2, 338, title, F(36, "bold"), TXT)
    ctext(d, W / 2, 394, sub, F(18, "light"), DIM)
    return hold(img, n)


def end_card(n):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    _logo(d, W / 2, 78, 50)
    rows = [("Agent collaboration", "planner -> 3 critics -> repair chair; perception verifies each action"),
            ("Multimodal intelligence", "Gemma-4 reads scene state from RoboCasa pixels"),
            ("Speed in action", "same model, 2.8x higher throughput than a GPU -> perception every action"),
            ("Physical AI", "real OSC execution, RoboCasa-native success, closed-loop recovery")]
    y = 196
    for a, b in rows:
        d.ellipse((188, y + 9, 202, y + 23), fill=ACC)
        d.text((222, y), a, font=F(23, "bold"), fill=TXT)
        d.text((222, y + 32), b, font=F(15, "light"), fill=DIM)
        y += 86
    return hold(img, n)


def caption_clip(frames, label, sub=""):
    out = []
    for im in frames:
        c = fit(im); d = ImageDraw.Draw(c)
        d.rounded_rectangle((24, H - 58, 24 + 12 * 0 + max(360, int(d.textlength(label, font=F(18))) + 40), H - 16),
                            8, fill=(0, 0, 0))
        d.text((38, H - 52), label, font=F(18, "bold"), fill=TXT)
        if sub:
            d.text((38, H - 30), sub, font=F(13, "regular"), fill=GOOD)
        out.append(c)
    return out


def speed_clip(n=150):
    sc = json.load(open(os.path.join(R, "speed_compare.json")))
    names = list(sc)
    cer = sc[names[0]]; gpu = sc[names[1]]
    tps_c, tps_g = cer["tok_per_s"], gpu["tok_per_s"]
    TOK = 420                                   # both generate the same plan (same model)
    time_c, time_g = TOK / tps_c, TOK / tps_g   # fair: time to emit the same #tokens
    span = time_g * 1.08
    ratio = tps_c / tps_g
    out = []
    for i in range(n):
        t = i / n * span
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        ctext(d, W / 2, 36, "SPEED IN ACTION", F(15, "bold"), ACC, track=3)
        ctext(d, W / 2, 60, "Same model (gemma-4-31b), same prompt", F(25, "bold"), TXT)
        ctext(d, W / 2, 96, "generating an identical plan — measuring only the serving hardware", F(14, "light"), DIM)
        lanes = [("Cerebras  ·  gemma-4-31b", tps_c, time_c, (47, 124, 255)),
                 ("Together GPU  ·  gemma-4-31b", tps_g, time_g, (150, 70, 80))]
        ly = 168
        for nm, tps, tot, c1 in lanes:
            d.text((90, ly - 30), nm, font=F(18, "bold"), fill=TXT)
            cur = min(t, tot)
            toks = int(min(t * tps, TOK))
            d.rounded_rectangle((90, ly, W - 90, ly + 54), 10, fill=(12, 18, 27), outline=(30, 42, 58))
            ww = int((W - 180) * (cur / span))
            if ww > 6:
                d.rounded_rectangle((90, ly, 90 + ww, ly + 54), 10, fill=c1)
            d.text((104, ly + 15), f"{toks:3d} tok", font=F(19, mono=True), fill=(235, 240, 248))
            if t >= tot:
                d.text((W - 256, ly + 15), f"{tot:.1f}s · {tps:.0f} tok/s", font=F(18, mono=True), fill=GOOD)
            ly += 132
        if t >= time_g:
            ctext(d, W / 2, 466, f"{ratio:.1f}× higher throughput on Cerebras  (252 vs 90 tok/s)", F(24, "bold"), GOOD)
            ctext(d, W / 2, 506, "fast enough to re-check perception after every robot action", F(16, "light"), DIM)
        out.append(img)
    return out


def build():
    seq = []
    seq += title_card(32)
    # how it works — agent architecture overview
    seq += load_still(os.path.join(R, "agents_spec.png"), 3.5)
    seq += section_card("1", "Closed-loop recovery", "Gemma catches a failed grasp from pixels -> replans -> succeeds", 14)
    REC = "/Users/yifankang/TAMPire/Screen Recording 2026-06-28 at 22.41.40.mov"
    seq += load_video(REC, 18.5, start=2.0)         # recovery dashboard: play -> native success
    seq += section_card("2", "Fixed-base long-horizon sort", "real OSC, base fixed, every grasp verified by Gemma", 14)
    PICKPLACE = "/Users/yifankang/TAMPire/pickplace.mov"
    seq += load_video(PICKPLACE, 16.0, start=2.0)   # sort dashboard: play -> 3/3 sorted
    seq += section_card("3", "Speed in action", "Gemma-4 on Cerebras vs a GPU provider", 14)
    seq += speed_clip(95)
    seq += section_card("4", "More real execution", "zero-shot mobile manipulation + long-horizon stacking", 14)
    seq += caption_clip(load_gif(os.path.join(R, "rc_vision_mobile.gif"), every=2)[:42],
                        "Zero-shot vision -> real drive -> grasp -> place", "RoboCasa native success")
    seq += caption_clip(load_gif(os.path.join(R, "tower_arm_n5.gif"), every=3)[:48],
                        "16-step long-horizon tower (disassemble -> rebuild)", "real Panda OSC · native success")
    seq += end_card(44)

    print(f"total {len(seq)} frames ≈ {len(seq)/FPS:.0f}s")
    out_mp4 = os.path.join(R, "TAMPire_demo_60s.mp4")
    try:
        import imageio
        imageio.mimsave(out_mp4, [np.array(s) for s in seq], fps=FPS, quality=8)
        print("wrote", out_mp4)
    except Exception as e:
        print("mp4 failed:", e)
    # lightweight gif preview
    sm = [s.resize((590, 300)).convert("P", palette=Image.ADAPTIVE, colors=128) for s in seq[::2]]
    gif = os.path.join(R, "TAMPire_demo_60s.gif")
    sm[0].save(gif, save_all=True, append_images=sm[1:], duration=int(2000 / FPS), loop=0, optimize=True)
    print("wrote", gif)


if __name__ == "__main__":
    build()
