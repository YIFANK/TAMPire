"""Composite a dashboard VIDEO from a closed-loop trace (frames + events):
left = robot camera, right = symbolic-state checkboxes + Gemma verdict log, top = speed.

    python -m tampire.tier2.make_dashboard_video --trace runs/trace_pickplace.json \
        --out runs/pickplace_dashboard
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (10, 14, 20); PANEL = (18, 24, 35); LINE = (30, 42, 58); TXT = (223, 231, 241)
DIM = (124, 138, 160); ACC = (78, 161, 255); GOOD = (55, 211, 154); BAD = (255, 93, 108)
WARN = (255, 200, 97); PURPLE = (183, 139, 255); BLUE2 = (127, 184, 255)
KIND_COLOR = {"perception": WARN, "plan": ACC, "action": BLUE2, "replan": WARN,
              "success": GOOD}


import os as _os
_LATO = _os.path.expanduser("~/Library/Fonts")
_FF = {"black": _LATO + "/Lato-Black.ttf", "bold": _LATO + "/Lato-Bold.ttf",
       "regular": _LATO + "/Lato-Regular.ttf", "light": _LATO + "/Lato-Light.ttf",
       "mono": "/System/Library/Fonts/SFNSMono.ttf"}


def _font(size, mono=False, weight="regular"):
    path = _FF["mono"] if mono else _FF.get(weight, _FF["regular"])
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _round_rect(d, box, r, fill=None, outline=None, w=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)


def build(trace_path, out_prefix):
    T = json.load(open(trace_path))
    frames = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in T["frames"]]
    ev = T["events"]
    mode = "preds" if any("preds" in e for e in ev) else "items"
    subtitle = T.get("subtitle", "")
    items = []
    for e in ev:
        m = re.search(r"holding\(([A-Za-z]+)\)", e["text"])
        if m and m.group(1) not in items and mode == "items":
            items.append(m.group(1))

    def goal_rows(revealed):
        """Return [(label, satisfied)] for the symbolic-goal panel."""
        if mode == "preds":
            p = {}
            for e in revealed:
                if "preds" in e:
                    p = e["preds"]
            return [("localized(object)", bool(revealed)),
                    ("holding(object)", p.get("holding", False)),
                    ("in(object, sink)", p.get("in_sink", False))]
        sat = set()
        for e in revealed:
            m = re.search(r"in_bin\(([A-Za-z]+)\)", e["text"])
            if m and e.get("ok"):
                sat.add(m.group(1))
        return [(f"in({it.lower()}, bin)", it in sat) for it in items]

    W, H = 1180, 600
    fbig = _font(26, weight="black"); ftag = _font(13, weight="light")
    fh = _font(12, weight="bold"); flog = _font(13, mono=True)
    fpred = _font(15, mono=True); fnum = _font(30, mono=True); fcap = _font(15, weight="bold")
    vid_w = 600
    out = []

    for fi in range(len(frames)):
        revealed = [e for e in ev if e["k"] <= fi]
        cur = revealed[-1] if revealed else None
        rows = goal_rows(revealed)
        placed = sum(1 for _, s in rows if s)

        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        # header
        d.text((28, 22), "TAMP", font=fbig, fill=TXT)
        d.text((28 + d.textlength("TAMP", font=fbig), 22), "ire", font=fbig, fill=ACC)
        d.text((30, 56), subtitle + " · Gemma-4 verifies from pixels", font=ftag, fill=DIM)
        # speed badge
        _round_rect(d, (W - 250, 20, W - 28, 70), 12, fill=(13, 30, 45), outline=(29, 77, 107))
        d.text((W - 234, 30), "ms / perception check", font=fh, fill=DIM)
        d.text((W - 234, 44), f"{T['metrics']['ms_each']}", font=fnum, fill=(127, 208, 255))
        d.text((W - 150, 52), "Gemma-4 · Cerebras", font=fh, fill=DIM)

        # video panel
        vx, vy = 28, 92
        fr = frames[fi]; scale = vid_w / fr.width; vh = int(fr.height * scale)
        fr2 = fr.resize((vid_w, vh))
        _round_rect(d, (vx - 4, vy - 4, vx + vid_w + 4, vy + vh + 4), 12, fill=(5, 8, 13))
        img.paste(fr2, (vx, vy))
        d.text((vx + 10, vy + 8), "● observation · robosuite PickPlace", font=fh, fill=(200, 210, 225))
        # caption
        if cur:
            cy = vy + vh - 38
            _round_rect(d, (vx + 8, cy, vx + vid_w - 8, vy + vh - 8), 8, fill=(0, 0, 0))
            col = KIND_COLOR.get(cur["kind"], ACC)
            if cur["kind"] == "check":
                col = GOOD if cur.get("ok") else BAD
            d.text((vx + 16, cy + 7), cur["kind"].upper(), font=fh, fill=col)
            d.text((vx + 16, cy + 20), cur["text"][:70], font=fcap, fill=TXT)

        # right column
        rx = vx + vid_w + 24; rw = W - rx - 28
        # symbolic state
        _round_rect(d, (rx, vy, rx + rw, vy + 168), 14, fill=PANEL, outline=LINE)
        d.text((rx + 14, vy + 12), "SYMBOLIC GOAL · VERIFIED FROM PIXELS", font=fh, fill=DIM)
        for k, (label, on) in enumerate(rows):
            iy = vy + 40 + k * 38
            _round_rect(d, (rx + 14, iy, rx + 32, iy + 18), 5,
                        fill=GOOD if on else None, outline=GOOD if on else (58, 74, 96), w=2)
            if on:
                d.text((rx + 18, iy + 1), "✓", font=fpred, fill=(10, 14, 20))
            d.text((rx + 42, iy), label, font=fpred, fill=GOOD if on else TXT)
        d.text((rx + rw - 90, vy + 12), f"{placed}/{len(rows)}", font=fh,
               fill=GOOD if placed == len(rows) else DIM)

        # log panel
        ly = vy + 184
        _round_rect(d, (rx, ly, rx + rw, vy + vh + 4), 14, fill=(7, 11, 17), outline=LINE)
        d.text((rx + 14, ly + 10), "GEMMA PERCEPTION · ACTION LOG", font=fh, fill=DIM)
        shown = revealed[-9:]
        yy = ly + 34
        for e in shown:
            col = KIND_COLOR.get(e["kind"], ACC)
            if e["kind"] == "check":
                col = GOOD if e.get("ok") else BAD
            tag = e["kind"][:5]
            d.text((rx + 14, yy), tag, font=fh, fill=col)
            txt = e["text"]
            if e.get("ms"):
                txt += f"  ·{e['ms']}ms"
            # wrap to width
            line = txt[:46]
            d.text((rx + 60, yy), line, font=flog, fill=(199, 210, 224))
            if len(txt) > 46:
                yy += 16; d.text((rx + 60, yy), txt[46:92], font=flog, fill=(199, 210, 224))
            yy += 22

        # footer status
        if any(e["kind"] == "success" for e in revealed):
            label = "✓ TASK COMPLETE — NATIVE SUCCESS" if mode == "preds" else f"✓ SORTED {placed}/{len(rows)} — COMPLETE"
            _round_rect(d, (vx, H - 40, vx + 320, H - 10), 8, fill=(14, 42, 32), outline=GOOD)
            d.text((vx + 14, H - 34), label, font=fcap, fill=GOOD)

        out.append(img)

    # write GIF
    gif = out_prefix + ".gif"
    sm = [im.convert("P", palette=Image.ADAPTIVE, colors=160) for im in out]
    sm[0].save(gif, save_all=True, append_images=sm[1:], duration=70, loop=0, optimize=True)
    print("wrote", gif, len(out), "frames")
    # try MP4
    try:
        import imageio
        mp4 = out_prefix + ".mp4"
        imageio.mimsave(mp4, [np.array(im) for im in out], fps=15, quality=8)
        print("wrote", mp4)
    except Exception as e:
        print("mp4 skipped:", str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="runs/trace_pickplace.json")
    ap.add_argument("--out", default="runs/pickplace_dashboard")
    a = ap.parse_args()
    build(a.trace, a.out)


if __name__ == "__main__":
    main()
