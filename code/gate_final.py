#!/usr/bin/env python3
"""논문 게이트 재현: 13특징 RF, 2초 인과 이동평균, 가동률 고정 3모델 비교

relmotion.py 와 gate3_v2.py 를 대체한다. 두 파일은 F1 이 최대가 되는
임계값을 고르고 binary_closing 과 최소지속 필터를 썼는데, 논문은 그
후처리를 인과 이동평균 하나로 대체했고 모델 비교는 가동률을 고정한다.
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ══════ 파라미터 ══════
W_REL     = 24        # 상대운동 특징 창 (2 s @ 12 Hz)
RADIUS    = 0.15      # 되돌아옴 판정 반경 (m)
W_SMOOTH  = 24        # 인과 이동평균 창 (2 s)
TH_RF     = 0.50      # RF 임계값
TARGET_DUTY = 0.56    # 세 모델을 맞출 가동률
N_LIDAR   = 2768
SAVE      = True
# ═════════════════════


def relfeat(C, frames):
    n=len(C); T=np.zeros((n,3))
    for i in range(n):
        lo=max(0,i-W_REL//2); hi=min(n,i+W_REL//2+1)
        w=np.arange(lo,hi)
        w=w[np.abs(frames[w]-frames[i])<=W_REL]
        if len(w)<6: continue
        P=C[w]
        path=np.linalg.norm(np.diff(P,axis=0),axis=1).sum()
        disp=np.linalg.norm(P[-1]-P[0])
        T[i,0]=disp
        T[i,1]=disp/max(path,1e-6)
        T[i,2]=(np.linalg.norm(P-P[len(P)//2],axis=1)<RADIUS).mean()
    return T


def causal_mean(x, w):
    """과거 w 프레임만 보는 후행 평균. 앞쪽은 첫 값으로 채움"""
    pad=np.concatenate([np.full(w-1, x[0]), x])
    return np.convolve(pad, np.ones(w)/w, mode='valid')


def threshold_for_duty(score, duty, valid):
    """유효구간 가동률이 목표에 가장 가까워지는 임계값"""
    best=None
    for th in np.arange(0.01, 1.00, 0.01):
        d=(score[valid]>th).mean()
        e=abs(d-duty)
        if best is None or e<best[0]: best=(e, th, d)
    return best[1], best[2]


# ── 학습 데이터 ──
X=np.load('features_X_v4.npy')
y=np.load('features_y_std.npy', allow_pickle=True)
fr=np.load('features_frame_v4.npy')
sc=np.load('features_scene_v4.npy', allow_pickle=True)
keep=X[:,3]>=0.25
X,y,fr,sc=X[keep],y[keep],fr[keep],sc[keep]
KEEP=np.load('feature_keep_idx.npy')

R=np.zeros((len(X),3))
for s in sorted(set(sc.tolist())):
    m=np.flatnonzero(sc==s); m=m[np.argsort(fr[m])]
    R[m]=relfeat(X[m][:,:3], fr[m])
Xa=np.hstack([X[:,KEEP], R])
yb=np.where(np.isin(y,['homi_work','squatting','squat_moving']),'target','other')
print(f'학습 {Xa.shape[0]}샘플  특징 {len(KEEP)} + 3 = {Xa.shape[1]}')

# ── S8 ──
X8=np.load('s8_feat_v4.npy'); F8=np.load('s8_frames_v4.npy')
o=np.argsort(F8); X8,F8=X8[o],F8[o]
X8a=np.hstack([X8[:,KEEP], relfeat(X8[:,:3], F8)])

STD=json.load(open('label_standard.json'))
gt=np.load('s8_groundtruth_v3.npy', allow_pickle=True)
t=np.isin(gt, STD['ground_work_gt']); NF=len(gt)
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') \
    else np.load('sync_model.npy')
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR)
print(f'S8 {NF}프레임 중 LiDAR 유효 {V.sum()}  기저 {t[V].mean()*100:.1f}%')

# ── RF 학습, 원시 확률 → 인과 평활 ──
rf=RandomForestClassifier(300, random_state=0, class_weight='balanced').fit(Xa, yb)
i=list(rf.classes_).index('target')
p=rf.predict_proba(X8a)[:,i]
pm={int(F8[k]): float(p[k]) for k in range(len(F8))}
raw=np.array([pm.get(int(li[c]), 0.0) for c in range(NF)])
score=causal_mean(raw, W_SMOOTH)
gate=score>TH_RF


def report(nm, g, th=None):
    tp=int((g&t&V).sum()); fp=int((g&~t&V).sum()); fn=int((~g&t&V).sum())
    P=tp/max(tp+fp,1); Rc=tp/max(tp+fn,1); F1=2*P*Rc/max(P+Rc,1e-9)
    d=g[V].mean()
    ts='  --' if th is None else f'{th:.2f}'
    print(f'{nm:16s} {ts}  P {P*100:5.1f}  R {Rc*100:5.1f}  '
          f'F1 {F1*100:5.1f}  duty {d*100:5.1f}')
    return dict(model=nm, threshold=th, precision=P, recall=Rc, f1=F1, duty=float(d))


print(f'\nAUC (평활 점수 기준) {roc_auc_score(t[V], score[V]):.3f}')
print('\n모델           Thr.  Prec.  Recall   F1    Duty')
res=[report('Random Forest', gate, TH_RF)]
res.append(report('Always on', np.ones(NF, bool)))

# ── 가동률 고정 비교 (다른 모델 점수가 있으면) ──
for nm, f in [('LSTM','s8_score_lstm_bin.npy'), ('PointNet','s8_score_pn_native.npy')]:
    if not os.path.exists(f):
        print(f'  ({f} 없음, {nm} 생략)'); continue
    s_=causal_mean(np.load(f), W_SMOOTH)
    th,_=threshold_for_duty(s_, TARGET_DUTY, V)
    res.append(report(nm, s_>th, th))

if SAVE:
    import joblib, os as _o
    _md=_o.path.join(_o.path.dirname(_o.getcwd()),'models')
    _o.makedirs(_md, exist_ok=True)
    joblib.dump({'model':rf,'keep_idx':KEEP,'w_rel':W_REL,'radius':RADIUS,
                 'w_smooth':W_SMOOTH,'threshold':TH_RF,
                 'classes':list(rf.classes_)},
                _o.path.join(_md,'gate_rf.joblib'), compress=3)
    print('저장: models/gate_rf.joblib')
    np.save('s8_score_raw.npy', raw)
    np.save('s8_score_final.npy', score)
    np.save('s8_gate_final_v2.npy', gate)
    json.dump({'w_rel':W_REL,'radius':RADIUS,'w_smooth':W_SMOOTH,
               'th_rf':TH_RF,'target_duty':TARGET_DUTY,
               'n_valid':int(V.sum()),'base_rate':float(t[V].mean()),
               'results':res},
              open('gate_final_results.json','w'), ensure_ascii=False, indent=2)
    print('\n저장: s8_score_raw.npy, s8_score_final.npy, '
          's8_gate_final_v2.npy, gate_final_results.json')
