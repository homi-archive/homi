#!/usr/bin/env python3
"""특징 조합과 인과 평활의 기여도. gate_final.py 와 같은 설정을 쓴다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os
from sklearn.ensemble import RandomForestClassifier

W_REL, RADIUS, W_SMOOTH = 24, 0.15, 24
TH, N_LIDAR = 0.50, 2768
TARGET_DUTY = 0.557          # 자세 단독 게이트의 가동률에 맞춰 비교


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
    return np.convolve(np.concatenate([np.full(w-1,x[0]),x]),
                       np.ones(w)/w, mode='valid')


X=np.load('features_X_v4.npy'); y=np.load('features_y_std.npy',allow_pickle=True)
fr=np.load('features_frame_v4.npy'); sc=np.load('features_scene_v4.npy',allow_pickle=True)
k=X[:,3]>=0.25; X,y,fr,sc=X[k],y[k],fr[k],sc[k]
KEEP=np.load('feature_keep_idx.npy')
R=np.zeros((len(X),3))
for s in sorted(set(sc.tolist())):
    m=np.flatnonzero(sc==s); m=m[np.argsort(fr[m])]
    R[m]=relfeat(X[m][:,:3],fr[m])
yb=np.where(np.isin(y,['homi_work','squatting','squat_moving']),'target','other')

X8=np.load('s8_feat_v4.npy'); F8=np.load('s8_frames_v4.npy')
o=np.argsort(F8); X8,F8=X8[o],F8[o]
R8=relfeat(X8[:,:3],F8)

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
POS=json.load(open('label_standard.json'))['ground_work_gt']
t=np.isin(gt,POS); NF=len(gt)
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR)

SETS=[('shape only (10)',        X[:,KEEP],                 X8[:,KEEP]),
      ('motion only (3)',        R,                         R8),
      ('shape + motion (13)',    np.hstack([X[:,KEEP],R]),  np.hstack([X8[:,KEEP],R8]))]


def score(XT,X8T):
    rf=RandomForestClassifier(300,random_state=0,class_weight='balanced').fit(XT,yb)
    i=list(rf.classes_).index('target')
    p=rf.predict_proba(X8T)[:,i]
    pm={int(F8[k]):float(p[k]) for k in range(len(F8))}
    return np.array([pm.get(int(li[c]),0.0) for c in range(NF)])


def PRF(g):
    tp=int((g&t&V).sum()); fp=int((g&~t&V).sum()); fn=int((~g&t&V).sum())
    P=tp/max(tp+fp,1); R_=tp/max(tp+fn,1)
    return P*100, R_*100, 2*P*R_/max(P+R_,1e-9)*100, g[V].mean()*100


def th_for_duty(s,d):
    return min(np.arange(0.01,1.00,0.005),
               key=lambda th: abs((s[V]>th).mean()-d))


rows=[]
print(f'{"features":22s} {"smoothing":10s} {"Thr":>5s} {"P":>6s} {"R":>6s} {"F1":>6s} {"duty":>6s}')
print('-'*66)
for nm,XT,X8T in SETS:
    raw=score(XT,X8T)
    for sm,ss in (('none',raw),(f'{W_SMOOTH/12:.0f} s causal',causal_mean(raw,W_SMOOTH))):
        th=TH if (sm!='none' and nm.startswith('shape +')) else th_for_duty(ss,TARGET_DUTY)
        P,R_,F,d=PRF(ss>th)
        print(f'{nm:22s} {sm:10s} {th:5.2f} {P:6.1f} {R_:6.1f} {F:6.1f} {d:6.1f}')
        rows.append(dict(features=nm,smoothing=sm,threshold=float(th),
                         precision=P,recall=R_,f1=F,duty=d))

# 클래스별 활성화율: 상대운동 특징이 무엇을 바꾸는가
CI=np.load('gate_s8_camera_idx.npy')
print('\n클래스별 활성화율 (가동률 고정, 2 s 평활)')
acts={}
for nm,XT,X8T in SETS:
    s=causal_mean(score(XT,X8T),W_SMOOTH)
    g=s>th_for_duty(s,TARGET_DUTY)
    acts[nm]={}
    for lab in sorted(set(gt[CI].tolist())):
        m=gt[CI]==lab
        acts[nm][lab]=float(g[CI][m].mean()*100)
labs=sorted(acts[SETS[0][0]], key=lambda l:-acts[SETS[-1][0]][l])
print(f'{"class":16s}'+''.join(f'{n[:12]:>14s}' for n,_,_ in SETS))
for lab in labs:
    print(f'{lab:16s}'+''.join(f'{acts[n][lab]:13.1f}%' for n,_,_ in SETS))

json.dump({'rows':rows,'activation':acts}, open('ablation_results.json','w'), indent=1)
print('\n저장: ablation_results.json')
