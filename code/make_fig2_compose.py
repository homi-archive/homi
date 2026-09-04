#!/usr/bin/env python3
"""Fig 2: 폴리캠 스캔 + 배치도, 가로 2패널 합성 + (a)(b) 라벨"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob, os, sys

SCAN_GLOB = "figures/scan_topdown.png"
SEARCH_DIRS = [".", "figures", "results_figures", "photos",
               os.path.expanduser("~/Downloads"),
               os.path.expanduser("~/Pictures")]
MAP      = "results_figures/fig2b_map.png"
OUT      = "results_figures/fig2_panels.png"

GUTTER, TRIM_PAD, PAD = 20, 8, 12
TRIM_MAP = True
COL_IN   = 3.5
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FR  = 0.055


def find_scan():
    for d in SEARCH_DIRS:
        hits = sorted(glob.glob(os.path.join(d, SCAN_GLOB)))
        hits = [h for h in hits if h.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if hits:
            print(f"스캔 발견: {hits[0]}")
            return hits[0]
    print(f"스캔을 못 찾음. 패턴 '{SCAN_GLOB}' 로 아래를 뒤졌음:")
    for d in SEARCH_DIRS:
        print("   ", os.path.abspath(d))
    print("파일 위치를 찾으려면:  find ~ -iname 'IMG_6945*'")
    print("찾은 경로를 SEARCH_DIRS 에 추가하거나 그 폴더로 복사할 것.")
    sys.exit(1)


def trim_white(im, pad=0, thr=250):
    a = np.array(im.convert("RGB"))
    ys, xs = np.where((a < thr).any(axis=2))
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(im.size[0], xs.max() + 1 + pad),
                    min(im.size[1], ys.max() + 1 + pad)))


scan = Image.open(find_scan()).convert("RGB")
if not os.path.exists(MAP):
    sys.exit(f"배치도가 없음: {MAP}\n먼저 python3 make_fig2b.py 를 실행할 것.")
fmap = Image.open(MAP).convert("RGB")
if TRIM_MAP:
    fmap = trim_white(fmap, TRIM_PAD)

H = fmap.size[1]
scan = scan.resize((round(scan.size[0] * H / scan.size[1]), H), Image.LANCZOS)

panels = [scan, fmap]
W = sum(p.size[0] for p in panels) + GUTTER * (len(panels) - 1)
canvas = Image.new("RGB", (W, H), "white")
x = 0
for p in panels:
    canvas.paste(p, (x, 0))
    x += p.size[0] + GUTTER

d = ImageDraw.Draw(canvas)
f = ImageFont.truetype(FONT_PATH, max(14, int(H * LABEL_FR)))
x = 0
for lab, p in zip(("(a)", "(b)"), panels):
    d.text((x + PAD, PAD), lab, fill=(30, 30, 30), font=f,
           stroke_width=3, stroke_fill=(255, 255, 255))
    x += p.size[0] + GUTTER

os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT)
print(f"저장  {OUT}  {canvas.size}")
print(f"  단 폭 {COL_IN} in 배치 시 높이 {H / W * COL_IN:.2f} in")
for name, p in zip(("scan", "map"), panels):
    inch = p.size[0] / W * COL_IN
    print(f"  {name:5s} {p.size[0]:5d} px -> {inch:.2f} in ({p.size[0] / inch:.0f} dpi)")

# ── 결과 열기 ──
import subprocess, shutil
VIEWER = shutil.which("eog") or shutil.which("xdg-open")
if VIEWER:
    subprocess.Popen([VIEWER, OUT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    print("뷰어를 못 찾음. 직접 열 것:", OUT)
