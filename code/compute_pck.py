#!/usr/bin/env python3
"""수동 주석 대비 YOLOv8n-pose 정확도.

척도는 같은 프레임에서 수동으로 찍은 상완 길이로 잡는다. 어깨 폭은 한쪽
팔만 보이는 프레임에서 정의되지 않아 표본을 크게 잃는다. 위치 오차와 함께
팔꿈치 각도 오차도 잰다. 6절이 쓰는 것은 위치가 아니라 각도이기 때문이다."""
import json, numpy as np

MAN  = json.load(open('keypoint_gt_s8.json'))
COCO = json.load(open('s8_coco17_full.json'))
IDX  = [5,6,7,8,9,10]
NM   = ['L shoulder','R shoulder','L elbow','R elbow','L wrist','R wrist']
CONF = 0.3
pred = {f['frame_idx']: f for f in COCO['frames']}

def ang(a,b,c):
    v1=np.array(a)-np.array(b); v2=np.array(c)-np.array(b)
    n=np.linalg.norm(v1)*np.linalg.norm(v2)
    if n<1e-6: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/n,-1,1))))

err=[[] for _ in range(6)]; dang={'L':[], 'R':[]}
nodet=noscale=0
for k,man in MAN.items():
    f=pred.get(int(k))
    if not f or not f['valid'] or f['kpts'] is None: nodet+=1; continue
    kp=f['kpts']
    ups=[]
    for s,e in ((0,2),(1,3)):
        if man[s] and man[e]:
            ups.append(np.linalg.norm(np.array(man[s])-np.array(man[e])))
    if not ups or max(ups)<5: noscale+=1; continue
    sc=float(np.median(ups))
    for j,i in enumerate(IDX):
        if man[j] is None or kp[i][2] <= CONF: continue
        err[j].append(np.linalg.norm(np.array(man[j])-np.array(kp[i][:2]))/sc)
    for side,(s,e,w) in (('L',(0,2,4)),('R',(1,3,5))):
        if all(man[q] is not None for q in (s,e,w)) and \
           all(kp[IDX[q]][2] > CONF for q in (s,e,w)):
            am=ang(man[s],man[e],man[w])
            ap=ang(kp[IDX[s]][:2],kp[IDX[e]][:2],kp[IDX[w]][:2])
            if am is not None and ap is not None:
                dang[side].append(abs(am-ap))

print(f'주석 {len(MAN)}프레임   검출 실패 {nodet}   척도 없음 {noscale}\n')
print(f'{"joint":12s}{"n":>5s}{"median":>9s}{"PCK@0.2":>9s}{"PCK@0.5":>9s}')
ALL=[]
for j in range(6):
    e=np.array(err[j]); ALL+=list(e)
    if not len(e): print(f'{NM[j]:12s}{0:5d}'); continue
    print(f'{NM[j]:12s}{len(e):5d}{np.median(e):9.3f}'
          f'{(e<=0.2).mean()*100:8.1f}%{(e<=0.5).mean()*100:8.1f}%')
A=np.array(ALL)
print(f'{"all":12s}{len(A):5d}{np.median(A):9.3f}'
      f'{(A<=0.2).mean()*100:8.1f}%{(A<=0.5).mean()*100:8.1f}%')

print('\n팔꿈치 각도 절대 오차 (도)')
for s in ('L','R'):
    d=np.array(dang[s])
    if not len(d): print(f'  {s}  표본 없음'); continue
    print(f'  {s}  n={len(d):3d}  중앙값 {np.median(d):5.1f}  '
          f'평균 {d.mean():5.1f}  10도 이내 {(d<=10).mean()*100:.0f}%  '
          f'20도 이내 {(d<=20).mean()*100:.0f}%')

json.dump({'n':int(len(A)),'median_norm_err':float(np.median(A)),
  'pck02':float((A<=0.2).mean()),'pck05':float((A<=0.5).mean()),
  'per_joint':{NM[j]:{'n':len(err[j]),
      'median':float(np.median(err[j])) if err[j] else None,
      'pck02':float((np.array(err[j])<=0.2).mean()) if err[j] else None}
      for j in range(6)},
  'elbow_angle_err_deg':{s:{'n':len(dang[s]),
      'median':float(np.median(dang[s])) if dang[s] else None,
      'within10':float((np.array(dang[s])<=10).mean()) if dang[s] else None}
      for s in ('L','R')}},
  open('pck_results.json','w'),indent=1)
print('\n저장: pck_results.json')
