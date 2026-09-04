#!/usr/bin/env python3
"""S8 궤적 좌표와 등방 배경 맵을 파일로 저장 (파이프라인 영상에서 재사용)"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2, json

# make_map_w6.py 의 구획 기하와 위치 추정을 그대로 사용
src = open('make_map_w6.py').read()
exec(src[:src.index("XR=(0.30,4.80)")], globals())

# ── 등방 배율 ──
XR = (0.30, 4.80)          # x: 화면 세로
YR = (-2.30, 3.30)         # y: 화면 가로
SPAN_X, SPAN_Y = XR[1]-XR[0], YR[1]-YR[0]
PPM = 130.0                # px per metre, 등방
MARGIN = 40

DRAW_W, DRAW_H = int(SPAN_Y*PPM), int(SPAN_X*PPM)
MW, MH = DRAW_W+2*MARGIN, DRAW_H+2*MARGIN
S = PPM

def T(x, y):
    return (int(MARGIN + (YR[1]-y)*S), int(MARGIN + (XR[1]-x)*S))

# ── 배경 맵 ──
BASE = np.full((MH, MW, 3), 255, np.uint8)
for f in FU+[F6]:
    cv2.rectangle(BASE, T(f['xhi'],f['yhi']), T(f['xlo'],f['ylo']), (245,245,245), -1)
for r in RG:
    cv2.rectangle(BASE, T(r['x1'],r['hi']), T(r['x0'],r['lo']), (222,222,222), -1)
    m = T((r['x0']+r['x1'])/2, (r['lo']+r['hi'])/2)
    (tw,th),_ = cv2.getTextSize(r['name'], cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.putText(BASE, r['name'], (m[0]-tw//2, m[1]+th//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140,140,140), 2, cv2.LINE_AA)
for f in FU:
    m = T((f['xlo']+f['xhi'])/2, (f['ylo']+f['yhi'])/2)
    (tw,th),_ = cv2.getTextSize(f['name'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(BASE, f['name'], (m[0]-tw//2, m[1]+th//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, cv2.LINE_AA)
m = T((F6['xlo']+F6['xhi'])/2, (YLO+YHI)/2)
(tw,th),_ = cv2.getTextSize('F6', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
cv2.putText(BASE, 'F6', (m[0]-tw//2, m[1]+th//2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, cv2.LINE_AA)
cv2.circle(BASE, T(*TANK), int(0.50*S), (240,240,240), -1)
cv2.putText(BASE, 'tank', (T(*TANK)[0]-14, T(*TANK)[1]+int(0.50*S)+16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1, cv2.LINE_AA)
LIDAR_MY = RG[1]['hi']
cv2.circle(BASE, T(0.36, LIDAR_MY), 6, (80,80,80), -1)
cv2.putText(BASE, 'LiDAR', (T(0.36,LIDAR_MY)[0]-18, T(0.36,LIDAR_MY)[1]+22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA)

# ── 궤적 저장 ──
N = C1-C0+1
track = np.full((N,2), np.nan, np.float32)
zones = []
for ci in range(C0, C1+1):
    st = pos.get(ci)
    if st is not None:
        track[ci-C0] = st[0]
        zones.append(st[1])
    else:
        zones.append('-')

np.save('track_s8.npy', track)
np.save('zones_s8.npy', np.array(zones))
cv2.imwrite('map_base.png', BASE)
json.dump({'C0':int(C0), 'C1':int(C1),
           'XR':list(XR), 'YR':list(YR),
           'PPM':PPM, 'MARGIN':MARGIN, 'MW':int(MW), 'MH':int(MH)},
          open('map_transform.json','w'), indent=1)

valid = int(np.isfinite(track[:,0]).sum())
print(f'저장 완료')
print(f'  track_s8.npy      {track.shape}  유효 {valid}/{N}')
print(f'  map_base.png      {MW}x{MH}  배율 {S:.1f} px/m (등방)')
print(f'  map_transform.json')
print(f'  그림 영역 {DRAW_W} x {DRAW_H} px,  가로세로비 {MW/MH:.3f}')
