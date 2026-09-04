#!/usr/bin/env python3
"""S8 전반에서 임계값을 고르고 후반에서만 평가한다.
   두 절반의 기저율이 다르므로 가동률이 아니라 F1 을 기준으로 고른다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os

W_SMOOTH, N_LIDAR = 24, 2768
MODELS = [('Random Forest', 's8_score_raw.npy'),
          ('LSTM',          's8_score_lstm_bin.npy'),
          ('PointNet',      's8_score_pn_native.npy')]

def causal_mean(x, w):
    return np.convolve(np.concatenate([np.full(w-1, x[0]), x]),
                       np.ones(w)/w, mode='valid')

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
POS=json.load(open('label_standard.json'))['ground_work_gt']
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
NF=len(gt); li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR); t=np.isin(gt[:NF],POS)
ci=np.flatnonzero(V); mid=ci[len(ci)//2]
A=V.copy(); A[mid:]=False
B=V.copy(); B[:mid]=False
print(f'분할 CAM {mid} (t={mid/10:.1f}s)')
print(f'  전반 {A.sum()}f  기저 {t[A].mean()*100:.1f}%')
print(f'  후반 {B.sum()}f  기저 {t[B].mean()*100:.1f}%\n')

def PRF(g,m):
    tp=int((g&t&m).sum()); fp=int((g&~t&m).sum()); fn=int((~g&t&m).sum())
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    return P*100,R*100,2*P*R/max(P+R,1e-9)*100,g[m].mean()*100

THS=np.arange(0.05,0.96,0.005)
print(f'{"model":15s} {"threshold from":16s} {"Thr":>5s} {"P":>6s} {"R":>6s} {"F1":>6s} {"duty":>6s}')
print('-'*68)
for m_ in (A,B):
    P,R,F,d=PRF(np.ones(NF,bool),m_)
    if m_ is B: print(f'{"Always on":15s} {"---":16s} {"--":>5s} {P:6.1f} {R:6.1f} {F:6.1f} {d:6.1f}')

rows=[]
for nm,f in MODELS:
    if not os.path.exists(f): print(f'  {f} 없음'); continue
    s=causal_mean(np.load(f)[:NF],W_SMOOTH)
    th_a=max(THS,key=lambda th:PRF(s>th,A)[2])   # 전반 F1 최대
    th_b=max(THS,key=lambda th:PRF(s>th,B)[2])   # 후반 오라클
    for tag,th in (('first half',th_a),('oracle (upper bound)',th_b)):
        P,R,F,d=PRF(s>th,B)
        print(f'{nm:15s} {tag:16s} {th:5.2f} {P:6.1f} {R:6.1f} {F:6.1f} {d:6.1f}')
        rows.append(dict(model=nm,selection=tag,threshold=float(th),
                         precision=P,recall=R,f1=F,duty=d))
    print(f'{"":15s} {"gap":16s} {"":5s} {"":6s} {"":6s} '
          f'{rows[-1]["f1"]-rows[-2]["f1"]:+6.1f}')

json.dump({'split_frame':int(mid),'base_first':float(t[A].mean()),
           'base_second':float(t[B].mean()),'rows':rows},
          open('split_half_results.json','w'),indent=1)
print('\n저장: split_half_results.json')
