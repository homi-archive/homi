#!/usr/bin/env python3
"""게이트 파이프라인 영상 (ground work 대상, 최종 설정)"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2, glob, os, json

a,b = np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') \
      else np.load('sync_model.npy')
gate = np.load('s8_gate_final_v2.npy')
try:
    _fur=np.load('furrow_s8.npy',allow_pickle=True)
    _fci=np.load('gate_s8_camera_idx.npy')
    FUR={int(c):str(z) for c,z in zip(_fci,_fur)}
    EXCLUDED_ZONE='F4'
    print(f'고랑 {len(FUR)}프레임 로드')
except FileNotFoundError:
    FUR={}; EXCLUDED_ZONE=None
    print('경고: furrow_s8.npy 없음, 자세 게이트만 사용')
gt   = np.load('s8_groundtruth_v3.npy', allow_pickle=True)
STD  = json.load(open('label_standard.json'))
LBL  = STD['paper_labels']
POS  = STD['ground_work_gt']
cam_all = sorted(glob.glob('video_S8_frames/*.jpg'))
# LiDAR 유효 구간만: lidar 0 ~ 2767 에 대응하는 카메라 프레임
C0 = int(np.ceil((0 - b)/a)); C1 = int(np.floor((2767.5 - b)/a))
C0 = max(C0,0); C1 = min(C1, len(cam_all)-1)
cam = cam_all[C0:C1+1]
print(f'LiDAR 유효 구간: CAM {C0}~{C1} ({len(cam)}프레임, {len(cam)/10:.1f}초)')

h0,w0 = cv2.imread(cam[0]).shape[:2]
H = 540; W = int(round(w0*H/h0)); PAD = 96
print(f'출력 {W*2}x{H+PAD}')
os.makedirs('results_figures', exist_ok=True)
out = cv2.VideoWriter('results_figures/gate_v4_S8.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (W*2, H+PAD))
blank = np.zeros((H,W,3), np.uint8)
NF = len(cam); ok = 0
cases = {'TP':[], 'FP':[], 'FN':[], 'TN':[]}

for ni, cf in enumerate(cam):
    ci = C0 + ni
    if ni % 300 == 0: print(f'  {ni}/{len(cam)}')
    li = int(round(a*ci + b))
    lp = f'vid_S8/f{li:05d}.png'
    c = cv2.resize(cv2.imread(cf), (W,H))
    if os.path.exists(lp):
        l0 = cv2.imread(lp); lh,lw = l0.shape[:2]
        s = min(W/lw, H/lh); r = cv2.resize(l0, (int(lw*s), int(lh*s)))
        l = blank.copy()
        y0=(H-r.shape[0])//2; x0=(W-r.shape[1])//2
        l[y0:y0+r.shape[0], x0:x0+r.shape[1]] = r; ok += 1
    else:
        l = blank.copy()
        cv2.putText(l,'no LiDAR',(W//2-90,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(90,90,90),2)

    posture = bool(gate[ci]) if ci < len(gate) else False
    zone = FUR.get(ci,'-')
    work_zone = (zone!='-') and (zone!=EXCLUDED_ZONE)
    on = posture and (work_zone if FUR else True)
    true = gt[ci] in POS
    if on and true:       col,tag = (90,255,140),'TP'
    elif on and not true: col,tag = (60,180,255),'FP'
    elif true:            col,tag = (80,80,255), 'FN'
    else:                 col,tag = (150,150,150),'TN'
    cases[tag].append(ci)

    if on:
        cv2.rectangle(c,(0,0),(W-1,H-1),col,5)
        cv2.rectangle(l,(0,0),(W-1,H-1),col,5)

    bar = np.zeros((PAD, W*2, 3), np.uint8)
    cv2.putText(bar,f'CAM {ci}   t={ci/10:6.1f}s',(16,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,255,255),2)
    cv2.putText(bar,f'annotated: {LBL.get(gt[ci],gt[ci])}',(16,56),
                cv2.FONT_HERSHEY_SIMPLEX,0.62,(210,225,255),2)
    cv2.putText(bar,f'LiDAR {li}',(W+16,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,255,255),2)
    cv2.putText(bar,f'gate {"ON " if on else "off"}   [{tag}]',(W+16,56),
                cv2.FONT_HERSHEY_SIMPLEX,0.68,col,2)
    if FUR:
        zc=(140,255,140) if work_zone else (200,150,90)
        cv2.putText(bar,f'furrow {zone}'+('' if zone=='-' else
                    (' work zone' if work_zone else ' excluded')),
                    (W+300,56),cv2.FONT_HERSHEY_SIMPLEX,0.58,zc,2)
    for x in range(W*2):
        i = C0 + int(x/(W*2)*len(cam))
        if i < NF:
            if gt[i] in POS: bar[PAD-16:PAD-10, x] = (190,190,190)
            if i < len(gate) and gate[i]: bar[PAD-8:PAD-2, x] = (90,255,140)
    px = int(ni/len(cam)*(W*2))
    cv2.line(bar,(px,PAD-18),(px,PAD),(0,255,255),2)
    cv2.putText(bar,'GT',(W*2-64,PAD-11),cv2.FONT_HERSHEY_SIMPLEX,0.32,(190,190,190),1)
    cv2.putText(bar,'gate',(W*2-64,PAD-3),cv2.FONT_HERSHEY_SIMPLEX,0.32,(90,255,140),1)
    out.write(np.vstack([np.hstack([c,l]), bar]))
out.release()
OUT='results_figures/gate_v4_S8.mp4'
print(f'완료: {OUT}  ({ok}/{NF})')
print(f'\n사례별 대표 프레임 (논문 그림용):')
for k,v in cases.items():
    if v: print(f'  {k}  {len(v):5d}개  대표 CAM {int(np.median(v))}')
json.dump({k:[int(x) for x in v] for k,v in cases.items()},
          open('gate_cases.json','w'))
print('저장: gate_cases.json')
import subprocess, shutil
V=shutil.which('vlc') or shutil.which('mpv') or shutil.which('xdg-open')
if V: subprocess.Popen([V,OUT],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
