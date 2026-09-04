#!/usr/bin/env python3
"""자세 게이트 / 위치 게이트 / 결합 비교 + 산출물 저장"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, collections
src=open('make_map_w6.py').read()
exec(src[:src.index("XR=(0.30,4.80)")], globals())
STD=json.load(open('label_standard.json')); POS=STD['ground_work_gt']
CI=np.arange(C0,C1+1)
t=np.isin(gt[C0:C1+1],POS)
gA=gate[C0:C1+1].astype(bool)                       # 자세 게이트
loc=np.array([pos[ci][1] if pos.get(ci) else '-' for ci in CI])
det=loc!='-'
def rep(nm,g):
    m=np.ones(len(CI),bool)
    tp=int((g&t&m).sum()); fp=int((g&~t&m).sum()); fn=int((~g&t&m).sum())
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    print(f'{nm:34s} P {P*100:5.1f}%  R {R*100:5.1f}%  '
          f'F1 {2*P*R/max(P+R,1e-9)*100:5.1f}%  duty {g.mean()*100:5.1f}%')
    return dict(name=nm,precision=P,recall=R,
                f1=2*P*R/max(P+R,1e-9),duty=float(g.mean()),
                tp=tp,fp=fp,fn=fn)
print(f'평가 {len(CI)}프레임   지면작업 기저 {t.mean()*100:.1f}%   '
      f'위치 검출 {det.mean()*100:.1f}%\n')
results=[]
results.append(rep('A. 자세만 (현재)', gA))
print()
for S in [['F1'],['F1','F2'],['F1','F2','F6'],['F1','F6']]:
    gB=np.isin(loc,S)
    results.append(rep(f'B. 위치만 {"+".join(S)}', gB))
print()
for S in [['F1'],['F1','F2'],['F1','F2','F6'],['F1','F6']]:
    gC=gA & np.isin(loc,S)
    results.append(rep(f'C. 자세 AND 위치 {"+".join(S)}', gC))
print()
gD=gA & (loc!='F4')
results.append(rep('D. 자세 AND (F4 제외)', gD))

# ── 산출물 저장 ──
np.save('furrow_s8.npy', loc)                # 프레임별 고랑 이름, 미검출은 '-'
np.save('gate_s8_posture.npy', gA)           # 자세 단독 게이트
np.save('gate_s8_noF4.npy', gD)              # F4 제외 결합 게이트
np.save('gate_s8_camera_idx.npy', CI)        # 위 세 배열의 카메라 프레임 번호

zone_counts=collections.Counter(loc.tolist())
json.dump({'camera_frame_range':[int(C0),int(C1)],
           'n_frames':int(len(CI)),
           'base_rate':float(t.mean()),
           'location_detected':float(det.mean()),
           'excluded_zone':'F4',
           'zone_counts':{k:int(v) for k,v in sorted(zone_counts.items())},
           'results':results},
          open('gate_location_results.json','w'),
          ensure_ascii=False, indent=2)

print('\n저장 완료')
print(f'  furrow_s8.npy            {loc.shape}  카메라 {C0}..{C1}')
print(f'  gate_s8_posture.npy      {gA.shape}')
print(f'  gate_s8_noF4.npy         {gD.shape}')
print(f'  gate_s8_camera_idx.npy   {CI.shape}')
print( '  gate_location_results.json')
print('\n구역별 프레임 수')
for k,v in sorted(zone_counts.items()):
    print(f'  {k:3s} {v:5d}  {v/len(CI)*100:5.1f}%')
