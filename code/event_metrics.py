#!/usr/bin/env python3
"""게이트의 사건 단위 성능. 지면작업이 긴 덩어리라 에피소드가 아니라
   중단(dropout)과 오작동(false activation)을 사건으로 센다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os

N_LIDAR = 2768
FPS     = 10.0
MIN_RUN = 5      # 0.5 s 미만 구간은 사건으로 세지 않음

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
g =np.load('s8_gate_final_v2.npy')
POS=json.load(open('label_standard.json'))['ground_work_gt']
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
NF=min(len(gt),len(g))
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR)
t=np.isin(gt[:NF],POS); gg=g[:NF].astype(bool)

def runs(mask, lo=1):
    d=np.flatnonzero(mask)
    if not len(d): return []
    st=[d[0]]+list(d[np.flatnonzero(np.diff(d)>1)+1])
    en=list(d[np.flatnonzero(np.diff(d)>1)])+[d[-1]]
    return [(s,e) for s,e in zip(st,en) if e-s+1>=lo]

dur=V.sum()/FPS
work=runs(t&V,1)
print(f'평가 {dur:.1f}s = {dur/60:.2f}분')
print(f'지면작업 {int((t&V).sum())}프레임, {len(work)}구간, '
      f'최장 {max(e-s+1 for s,e in work)/FPS:.1f}s\n')

# ── 작업 중 게이트가 닫히는 사건 ──
drop=runs(t&~gg&V, MIN_RUN)
dl=np.array([(e-s+1)/FPS for s,e in drop])
print(f'작업 중 중단 (게이트가 {MIN_RUN/FPS:.1f}s 이상 닫힘)')
print(f'  {len(drop)}회 = 분당 {len(drop)/(dur/60):.2f}회')
print(f'  총 {dl.sum():.1f}s, 중앙값 {np.median(dl):.1f}s, 최장 {dl.max():.1f}s')
print(f'  2 s 이내 회복 {int((dl<=2.0).sum())}/{len(dl)} = {(dl<=2.0).mean()*100:.0f}%')

# ── 오작동 ──
fa=runs(gg&~t&V, MIN_RUN)
fl=np.array([(e-s+1)/FPS for s,e in fa])
print(f'\n오작동 (비작업 중 게이트가 {MIN_RUN/FPS:.1f}s 이상 열림)')
print(f'  {len(fa)}회 = 분당 {len(fa)/(dur/60):.2f}회')
print(f'  총 {fl.sum():.1f}s, 중앙값 {np.median(fl):.1f}s, 최장 {fl.max():.1f}s')

# ── 전환 지연 ──
def edges(cond):
    d=np.flatnonzero(cond & V)
    return [x for x in d if x>0 and not (cond[x-1] and V[x-1])]

lat=[]
for x in edges(t):                       # 작업 시작 시점
    j=x
    while j<NF and V[j] and not gg[j]: j+=1
    if j<NF and V[j]: lat.append((j-x)/FPS)
rel=[]
for x in edges(~t):                      # 작업 종료 시점
    j=x
    while j<NF and V[j] and gg[j]: j+=1
    if j<NF and V[j]: rel.append((j-x)/FPS)
lat=np.array(lat); rel=np.array(rel)
print(f'\n작업 시작 -> 게이트 열림 ({len(lat)}회)')
print(f'  중앙값 {np.median(lat):.2f}s  평균 {lat.mean():.2f}s  최대 {lat.max():.2f}s')
print(f'  2 s 이내 {int((lat<=2.0).sum())}/{len(lat)} = {(lat<=2.0).mean()*100:.0f}%')
print(f'작업 종료 -> 게이트 닫힘 ({len(rel)}회)')
print(f'  중앙값 {np.median(rel):.2f}s  평균 {rel.mean():.2f}s  최대 {rel.max():.2f}s')
print(f'  2 s 이내 {int((rel<=2.0).sum())}/{len(rel)} = {(rel<=2.0).mean()*100:.0f}%')

json.dump({'eval_seconds':dur,
  'dropouts':len(drop),'dropouts_per_min':len(drop)/(dur/60),
  'dropout_median_s':float(np.median(dl)),'dropout_max_s':float(dl.max()),
  'dropout_total_s':float(dl.sum()),
  'false_activations':len(fa),'fa_per_min':len(fa)/(dur/60),
  'fa_median_s':float(np.median(fl)),'fa_max_s':float(fl.max()),
  'fa_total_s':float(fl.sum()),
  'onset_median_s':float(np.median(lat)),'onset_max_s':float(lat.max()),
  'onset_within_2s':float((lat<=2.0).mean()),
  'release_median_s':float(np.median(rel)),'release_max_s':float(rel.max())},
  open('event_metrics.json','w'),indent=1)
print('\n저장: event_metrics.json')
