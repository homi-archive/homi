#!/usr/bin/env python3
"""위치 특징 누출: 중심 좌표를 넣으면 교차검증은 오르고 held-out 은 무너진다.
   gate_final.py 와 같은 규약(13특징, 2초 인과평활, 0.50 임계)을 쓴다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

W_REL, RADIUS, W_SMOOTH, TH, N_LIDAR = 24, 0.15, 24, 0.50, 2768

def relfeat(C, frames):
    n=len(C); T=np.zeros((n,3))
    for i in range(n):
        lo=max(0,i-W_REL//2); hi=min(n,i+W_REL//2+1)
        w=np.arange(lo,hi); w=w[np.abs(frames[w]-frames[i])<=W_REL]
        if len(w)<6: continue
        P=C[w]
        path=np.linalg.norm(np.diff(P,axis=0),axis=1).sum()
        disp=np.linalg.norm(P[-1]-P[0])
        T[i,0]=disp; T[i,1]=disp/max(path,1e-6)
        T[i,2]=(np.linalg.norm(P-P[len(P)//2],axis=1)<RADIUS).mean()
    return T

def causal_mean(x,w):
    return np.convolve(np.concatenate([np.full(w-1,x[0]),x]),np.ones(w)/w,mode='valid')

X=np.load('features_X_v4.npy'); y6=np.load('features_y_std.npy',allow_pickle=True)
ystd=np.load('features_y_std.npy',allow_pickle=True)
fr=np.load('features_frame_v4.npy'); sc=np.load('features_scene_v4.npy',allow_pickle=True)
k=X[:,3]>=0.25; X,y6,ystd,fr,sc=X[k],y6[k],ystd[k],fr[k],sc[k]
KEEP=list(np.load('feature_keep_idx.npy'))

R=np.zeros((len(X),3))
for s in sorted(set(sc.tolist())):
    m=np.flatnonzero(sc==s); m=m[np.argsort(fr[m])]
    R[m]=relfeat(X[m][:,:3],fr[m])
yb=np.where(np.isin(ystd,['homi_work','squatting','squat_moving']),'target','other')

X8=np.load('s8_feat_v4.npy'); F8=np.load('s8_frames_v4.npy')
o=np.argsort(F8); X8,F8=X8[o],F8[o]
R8=relfeat(X8[:,:3],F8)

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
POS=json.load(open('label_standard.json'))['ground_work_gt']
t=np.isin(gt,POS); NF=len(gt)
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR)

def run(cols, nm):
    XT=np.hstack([X[:,cols],R]); X8T=np.hstack([X8[:,cols],R8])
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    acc=cross_val_score(RandomForestClassifier(200,random_state=0,
        class_weight='balanced'), X[:,cols], y6, cv=cv, n_jobs=-1).mean()*100
    rf=RandomForestClassifier(300,random_state=0,class_weight='balanced').fit(XT,yb)
    i=list(rf.classes_).index('target')
    p=rf.predict_proba(X8T)[:,i]
    pm={int(F8[j]):float(p[j]) for j in range(len(F8))}
    s=causal_mean(np.array([pm.get(int(li[c]),0.0) for c in range(NF)]),W_SMOOTH)
    g=s>TH
    tp=int((g&t&V).sum()); fp=int((g&~t&V).sum()); fn=int((~g&t&V).sum())
    P=tp/max(tp+fp,1); Rc=tp/max(tp+fn,1)
    F1=2*P*Rc/max(P+Rc,1e-9)*100
    print(f'{nm:34s} 6-class CV {acc:5.1f}%   held-out gate F1 {F1:5.1f}%')
    return acc, F1

print(f'형상 특징 열: {KEEP}\n')
a0,f0 = run(KEEP,            'shape only (paper)')
a1,f1 = run(KEEP+[0],        '+ x centroid')
a2,f2 = run(KEEP+[0,1],      '+ x,y centroid')
a3,f3 = run(KEEP+[0,1,2],    '+ x,y,z centroid')
print(f'\n논문 문장용:')
print(f'  CV {a0:.1f} -> {a3:.1f}   held-out F1 {f0:.1f} -> {f3:.1f}')
print(f'  x 좌표만 추가: F1 {f0:.1f} -> {f1:.1f}  ({f0-f1:+.1f})')
json.dump({'shape_only':[a0,f0],'plus_x':[a1,f1],
           'plus_xy':[a2,f2],'plus_xyz':[a3,f3]},
          open('position_leakage_results.json','w'),indent=1)
print('저장: position_leakage_results.json')
