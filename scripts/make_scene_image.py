"""Generate a simple tabletop image for the vision demo: red/green/blue blocks
and a bowl on a wood-toned table. No assets needed."""
from __future__ import annotations

import sys

from PIL import Image, ImageDraw


def make(path: str = "scenes/scene.png", w: int = 512, h: int = 384) -> str:
    img = Image.new("RGB", (w, h), (228, 228, 232))   # wall
    d = ImageDraw.Draw(img)
    # table surface
    d.rectangle([0, int(h * 0.45), w, h], fill=(196, 158, 110))
    d.rectangle([0, int(h * 0.45), w, int(h * 0.47)], fill=(160, 120, 78))

    def block(cx, cy, s, color):
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color, outline=(30, 30, 30), width=2)

    # red block with a green block stacked on top (matches blocks.json trap)
    block(150, 300, 26, (200, 40, 40))
    block(150, 252, 22, (40, 160, 60))
    # standalone blue block
    block(300, 300, 26, (45, 80, 200))
    # bowl (ellipse) on the right
    d.ellipse([380, 280, 470, 330], fill=(120, 150, 210), outline=(40, 40, 40), width=3)
    d.ellipse([395, 285, 455, 315], fill=(90, 120, 180))

    img.save(path)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "scenes/scene.png"
    print("wrote", make(out))
