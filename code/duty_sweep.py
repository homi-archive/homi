#!/usr/bin/env python3
"""가동률 대비 게이트 성능 곡선. 임계값을 테스트 세션에서 고르지 않았음을 보인다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os, subprocess, shutil
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

W_SMOOTH  = 24
N_LIDAR   = 2768
DUTY_LO, DUTY_HI = 0.30, 0.85     # 곡선을 그릴 가동률 범위
BAND_LO, BAND_HI = 0.40, 0.70     # 본문에서 주장할 구간
OUT_PNG = 'results_figures/duty_sweep.png'
OUT_PDF = 'results_figures/duty_sweep.pdf'

MODELS = [('Random Forest', 's8_score_raw.npy',      '#1a4f8a', '-'),
          ('LSTM',          's8_score_lstm_bin.npy', '#c05a1a', '--'),
          ('PointNet',      's8_score_pn_native.npy',   '#4a7a3a', ':')]


def causal_mean(x, w):
    return np.convolve(np.concatenate([np.full(w-1, x[0]), x]),
                       np.ones(w)/w, mode='valid')


gt = np.load('s8_groundtruth_v3.npy', allow_pickle=True)
POS = json.load(open('label_standard.json'))['ground_work_gt']
a, b = np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') \
       else np.load('sync_model.npy')
NF = len(gt)
li = np.array([int(round(a*c+b)) for c in range(NF)])
V = (li >= 0) & (li < N_LIDAR)
t = np.isin(gt[:NF], POS)
print(f'유효 {V.sum()}프레임   기저 {t[V].mean()*100:.1f}%')

curves = {}
for nm, f, _, _ in MODELS:
    if not os.path.exists(f):
        print(f'  {f} 없음, {nm} 생략'); continue
    s = causal_mean(np.load(f)[:NF], W_SMOOTH)
    rows = []
    for th in np.arange(0.01, 1.00, 0.005):
        g = s > th
        d = g[V].mean()
        if not (DUTY_LO <= d <= DUTY_HI): continue
        tp = int((g & t & V).sum()); fp = int((g & ~t & V).sum())
        fn = int((~g & t & V).sum())
        P = tp/max(tp+fp, 1); R = tp/max(tp+fn, 1)
        rows.append((d, P, R, 2*P*R/max(P+R, 1e-9), th))
    rows.sort()
    curves[nm] = np.array(rows)
    print(f'  {nm:14s} {len(rows)}점  duty {rows[0][0]*100:.0f}-{rows[-1][0]*100:.0f}%')

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
for nm, _, c, ls in MODELS:
    if nm not in curves: continue
    q = curves[nm]
    ax[0].plot(q[:,0]*100, q[:,3]*100, color=c, ls=ls, lw=1.6, label=nm)
    ax[1].plot(q[:,0]*100, q[:,1]*100, color=c, ls=ls, lw=1.6, label=nm)

for A, yl in zip(ax, ('F1 (%)', 'precision (%)')):
    A.axvspan(BAND_LO*100, BAND_HI*100, color='#000000', alpha=.05, lw=0)
    A.set_xlabel('camera duty cycle (%)', fontsize=8)
    A.set_ylabel(yl, fontsize=8)
    A.tick_params(labelsize=8)
    A.grid(alpha=.25, lw=.5)
    for sp in ('top','right'): A.spines[sp].set_visible(False)
ax[0].legend(fontsize=7.5, frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')

# ── 본문용 수치 ──
print(f'\n{BAND_LO*100:.0f}-{BAND_HI*100:.0f}% 구간 요약')
summary = {}
for nm in curves:
    q = curves[nm]
    m = (q[:,0] >= BAND_LO) & (q[:,0] <= BAND_HI)
    if not m.any(): continue
    summary[nm] = dict(f1_mean=float(q[m,3].mean()), f1_max=float(q[m,3].max()),
                       f1_min=float(q[m,3].min()),
                       prec_mean=float(q[m,1].mean()))
    print(f'  {nm:14s} F1 평균 {q[m,3].mean()*100:5.1f}  '
          f'범위 {q[m,3].min()*100:5.1f}-{q[m,3].max()*100:5.1f}  '
          f'정밀도 평균 {q[m,1].mean()*100:5.1f}')

if len(summary) > 1:
    names = list(summary)
    best = max(names, key=lambda n: summary[n]['f1_mean'])
    print(f'\n{BAND_LO*100:.0f}-{BAND_HI*100:.0f}% 전 구간에서 최고인 모델 확인')
    ref = curves[best]
    always = True
    for nm in names:
        if nm == best: continue
        q = curves[nm]
        for d, P, R, F, th in q:
            if not (BAND_LO <= d <= BAND_HI): continue
            j = np.argmin(np.abs(ref[:,0]-d))
            if ref[j,3] <= F: always = False; break
    print(f'  {best} 가 {"모든" if always else "대부분"} 가동률에서 우위')

json.dump({'band':[BAND_LO,BAND_HI],
           'summary':summary,
           'curves':{k:v.tolist() for k,v in curves.items()}},
          open('duty_sweep_results.json','w'), indent=1)
print(f'\n저장: {OUT_PNG}, {OUT_PDF}, duty_sweep_results.json')

V_ = shutil.which('eog') or shutil.which('xdg-open')
if V_: subprocess.Popen([V_, OUT_PNG],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
