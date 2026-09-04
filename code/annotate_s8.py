#!/usr/bin/env python3
"""S8 키보드 주석 도구. 영상만 보고 라벨을 찍는다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2, glob, os, json, collections

OUT    = 's8_annotation_v4.json'
NPY    = 's8_groundtruth_v4.npy'
FRAMES = 'video_S8_frames/*.jpg'
NF     = 2498
DISP_W = 1080
STEP_AFTER_LABEL = 10      # 라벨 찍으면 이만큼 자동 전진

KEYS = [('1','homi_work',(60,150,70)), ('2','handweed',(90,180,110)),
        ('3','squatting',(120,200,140)), ('4','squat_moving',(150,220,170)),
        ('5','stooped_move',(180,140,60)), ('6','walking',(190,120,120)),
        ('7','standing',(150,150,150)), ('8','other',(60,110,200)),
        ('9','watering',(200,160,60))]
KEYMAP = {ord(k): v for k, v, _ in KEYS}
COLOR  = {v: c for _, v, c in KEYS}

cam = sorted(glob.glob(FRAMES))
assert len(cam) >= NF, f'프레임 {len(cam)}개, {NF} 필요'

marks = {}
if os.path.exists(OUT):
    marks = {int(k): v for k, v in json.load(open(OUT))['marks'].items()}
    print(f'{OUT} 이어서 작업. 경계 {len(marks)}개')


def resolve():
    lab = np.array([''] * NF, dtype=object); cur = ''
    for f in range(NF):
        if f in marks: cur = marks[f]
        lab[f] = cur
    return lab


def save(lab):
    json.dump({'marks': {str(k): v for k, v in sorted(marks.items())},
               'n_frames': NF}, open(OUT, 'w'), ensure_ascii=False, indent=1)
    np.save(NPY, lab)


def panel(lab, i):
    im = cv2.imread(cam[i])
    im = cv2.resize(im, (DISP_W, int(im.shape[0] * DISP_W / im.shape[1])))

    hdr = np.full((78, DISP_W, 3), 255, np.uint8)
    cv2.putText(hdr, f'CAM {i}   t={i/10:.1f}s', (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2, cv2.LINE_AA)
    cur = lab[i] if lab[i] else '(unset)'
    cv2.putText(hdr, cur, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, .9,
                COLOR.get(lab[i], (40, 40, 40)), 2, cv2.LINE_AA)
    if i in marks:
        cv2.putText(hdr, 'MARK', (DISP_W - 430, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (0, 0, 220), 2, cv2.LINE_AA)
    nun = int(sum(1 for f in range(NF) if not lab[f]))
    cv2.putText(hdr, f'unset {nun}   marks {len(marks)}', (DISP_W - 330, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(hdr, 'a/d 1f   w/s 10f   ,/. 50f   [/] 200f   g jump'
                     '   u undo   x unmark   p list   q quit',
                (12, DISP_W and 78 - 4), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (110, 110, 110), 1, cv2.LINE_AA)

    # 라벨 키 안내
    leg = np.full((34, DISP_W, 3), 255, np.uint8)
    x = 10
    for k, v, c in KEYS:
        cv2.rectangle(leg, (x, 8), (x + 16, 24), c, -1)
        cv2.putText(leg, f'{k} {v}', (x + 21, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (40, 40, 40), 1, cv2.LINE_AA)
        x += 24 + int(cv2.getTextSize(f'{k} {v}', cv2.FONT_HERSHEY_SIMPLEX,
                                      .45, 1)[0][0]) + 14

    tl = np.full((30, DISP_W, 3), 245, np.uint8)
    for px in range(DISP_W):
        c = COLOR.get(lab[int(px / DISP_W * NF)])
        if c: tl[:, px] = c
    px = int(i / NF * DISP_W)
    cv2.line(tl, (px, 0), (px, 29), (0, 0, 0), 2)
    return np.vstack([hdr, leg, im, tl])


i = 0
print('a/d 1프레임  w/s 10  ,/. 50  [/] 200  숫자 라벨  q 종료')
while True:
    lab = resolve()
    cv2.imshow('S8', panel(lab, i))
    k = cv2.waitKeyEx(0)

    if k in (ord('q'), 27): break
    elif k in (ord('d'), 65363, 83): i = min(i + 1, NF - 1)
    elif k in (ord('a'), 65361, 81): i = max(i - 1, 0)
    elif k == ord('s'): i = min(i + 10, NF - 1)
    elif k == ord('w'): i = max(i - 10, 0)
    elif k == ord('.'): i = min(i + 50, NF - 1)
    elif k == ord(','): i = max(i - 50, 0)
    elif k == ord(']'): i = min(i + 200, NF - 1)
    elif k == ord('['): i = max(i - 200, 0)
    elif k in KEYMAP:
        marks[i] = KEYMAP[k]
        print(f'  CAM {i} -> {KEYMAP[k]}')
        save(resolve())
        i = min(i + STEP_AFTER_LABEL, NF - 1)
    elif k == ord('x'):
        if i in marks:
            print(f'  CAM {i} 경계 삭제 ({marks.pop(i)})'); save(resolve())
    elif k == ord('u'):
        if marks:
            f = max(marks); print(f'  CAM {f} 취소 ({marks.pop(f)})')
            i = f; save(resolve())
    elif k == ord('g'):
        try: i = max(0, min(int(input('프레임 번호: ')), NF - 1))
        except ValueError: pass
    elif k == ord('p'):
        lab = resolve(); print('\n--- 구간 ---'); s = 0
        for f in range(1, NF + 1):
            if f == NF or lab[f] != lab[s]:
                if lab[s]: print(f'  {s:5d}-{f-1:5d}  {f-s:5d}f  {lab[s]}')
                s = f
        print()

cv2.destroyAllWindows()
lab = resolve(); save(lab)
print(f'\n저장: {OUT}, {NPY}')
nun = int(sum(1 for f in range(NF) if not lab[f]))
if nun: print(f'  경고: 라벨 없는 프레임 {nun}개')
for k, v in sorted(collections.Counter(lab.tolist()).items(), key=lambda t: -t[1]):
    if k: print(f'  {k:14s} {v:5d}프레임 {v/10:7.1f}s')
