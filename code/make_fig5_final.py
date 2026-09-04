#!/usr/bin/env python3
"""Fig 5: 게이트 타임라인 + 점유 고랑 레인 + 누적 동선 맵 (전폭)"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, os, subprocess, shutil
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

FURROW_FILE = 'furrow_s8.npy'
CAMIDX_FILE = 'gate_s8_camera_idx.npy'
TRACK_FILE  = 'track_s8.npy'
MAPBASE     = 'map_base.png'
MAPTF       = 'map_transform.json'
ZONES     = ['F1','F2','F3','F4','F5','F6']
EXCLUDED  = 'F4'

# ── 색 ──
C_ANNOT   = '#1a1a1a'
C_GATE    = '#8a8a8a'
C_FILL    = '#5b8fc9'
C_LINE    = '#1a4f8a'
C_FP      = '#ef6c00'
C_FN      = '#c62828'
C_NOLIDAR = '#9e9e9e'
BASE      = ['#E8DCCB','#D8C3A5','#C7AA85','#B59166','#A37A4C','#8F6437']
EXC_COLOR = '#4C6B8A'
UNKNOWN   = '#F4F4F4'
N_LIDAR_S8= 2768

# ── 맵 누적 ──
TRAIL_A  = 0.004      # 프레임당 누적 강도 (영상과 동일)
DOT_R_M  = 0.20       # 누적 원 반지름 (m)
MAP_FRAC = 0.30       # 전체 폭에서 맵이 차지하는 비율

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
g =np.load('s8_gate_final_v2.npy')
s =np.load('s8_score_final.npy')
STD=json.load(open('label_standard.json')); POS=STD['ground_work_gt']
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
NF=min(len(gt),len(g)); t=np.arange(NF)/10.0
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR_S8)
tgt=np.isin(gt[:NF],POS); gg=g[:NF]

fur=None; present=[]
if os.path.exists(FURROW_FILE) and os.path.exists(CAMIDX_FILE):
    raw=np.load(FURROW_FILE,allow_pickle=True); CI=np.load(CAMIDX_FILE)
    lut={z:i for i,z in enumerate(ZONES)}
    idx=np.array([lut.get(str(v),-1) for v in raw],dtype=int)
    fur=np.full(NF,-1,int); m=(CI>=0)&(CI<NF)
    fur[CI[m]]=idx[m]; fur[~V]=-1
    present=[z for i,z in enumerate(ZONES) if (fur==i).any()]
    print(f'고랑 {len(raw)}프레임, CAM {CI[0]}..{CI[-1]}, 등장 {present}')

# ── 누적 동선 맵 (세션 종료 시점) ──
mapimg=None
if all(os.path.exists(f) for f in (TRACK_FILE,MAPBASE,MAPTF)):
    import cv2
    MB=cv2.cvtColor(cv2.imread(MAPBASE),cv2.COLOR_BGR2RGB)
    TR=np.load(TRACK_FILE)
    MT=json.load(open(MAPTF))
    MXR,MYR,MPPM,MM=MT['XR'],MT['YR'],MT['PPM'],MT['MARGIN']
    Hm,Wm=MB.shape[:2]
    def Tm(x,y):
        return (int(MM+(MYR[1]-y)*MPPM), int(MM+(MXR[1]-x)*MPPM))
    ACC=np.zeros((Hm,Wm),np.float32)
    R=max(6,int(DOT_R_M*MPPM))
    for k in range(len(TR)):
        if not np.isfinite(TR[k,0]): continue
        lay=np.zeros((Hm,Wm),np.float32)
        cv2.circle(lay,Tm(*TR[k]),R,1.0,-1,cv2.LINE_AA)
        ACC=np.clip(ACC+lay*TRAIL_A,0,1.0)
    LO=np.array([180,205,245],np.float32)   # RGB 연한 파랑
    HI=np.array([ 20, 40,200],np.float32)   # RGB 진한 파랑
    c3=ACC[...,None]
    mapimg=(MB.astype(np.float32)*(1-c3)+(LO*(1-c3)+HI*c3)*c3).astype(np.uint8)
    print(f'누적 맵 {Wm}x{Hm}, 최대 누적 {ACC.max():.2f}, '
          f'유효 {int(np.isfinite(TR[:,0]).sum())}프레임')
else:
    print('경고: 맵 파일 없음. save_track_s8.py 를 먼저 실행할 것')

# ── 도화 ──
nrow=3 if fur is not None else 2
hr  =[1.0,2.0,0.5] if fur is not None else [1.0,2.0]
FIGW=13.0 if mapimg is not None else 9.4
fig=plt.figure(figsize=(FIGW,3.9 if fur is not None else 3.4))
if mapimg is not None:
    gs=fig.add_gridspec(1,2,width_ratios=[1-MAP_FRAC,MAP_FRAC],wspace=0.14,
                    left=0.045,right=0.985)
    gsl=gs[0].subgridspec(nrow,1,height_ratios=hr,hspace=0.22)
    ax=[fig.add_subplot(gsl[i]) for i in range(nrow)]
    for i in range(1,nrow): ax[i].sharex(ax[0])
    axm=fig.add_subplot(gs[1])
else:
    gsl=fig.add_gridspec(nrow,1,height_ratios=hr,hspace=0.22)
    ax=[fig.add_subplot(gsl[i]) for i in range(nrow)]
    for i in range(1,nrow): ax[i].sharex(ax[0])
    axm=None

for y,(arr,c) in enumerate([(tgt&V,C_ANNOT),(gg&V,C_GATE)]):
    prev=arr[0]; st=0
    for i in range(1,NF+1):
        if i==NF or arr[i]!=prev:
            if prev: ax[0].barh(y,(i-st)/10,left=st/10,height=0.62,color=c,lw=0)
            if i<NF: prev=arr[i]; st=i
ax[0].set_yticks([0,1]); ax[0].set_yticklabels(['annotated','gate'],fontsize=8)
ax[0].set_ylim(-0.5,1.5); ax[0].invert_yaxis()
ax[0].tick_params(labelsize=8,labelbottom=False); ax[0].set_xlim(0,NF/10)
for sp in ('top','right','left'): ax[0].spines[sp].set_visible(False)

ax[1].fill_between(t,0,s[:NF],color=C_FILL,alpha=0.6,lw=0)
ax[1].plot(t,s[:NF],color=C_LINE,lw=0.6)
ax[1].axhline(0.50,color=C_FN,ls='--',lw=1.0)
ax[1].text(NF/10*1.012,0.50,'0.50',fontsize=7.5,color=C_FN,va='center',ha='left')
fp=gg&~tgt&V; fn=(~gg)&tgt&V
for arr,c in [(fp,C_FP),(fn,C_FN)]:
    prev=arr[0]; st=0
    for i in range(1,NF+1):
        if i==NF or arr[i]!=prev:
            if prev: ax[1].axvspan(st/10,i/10,color=c,alpha=0.28,lw=0)
            if i<NF: prev=arr[i]; st=i
ax[1].set_ylim(0,1); ax[1].set_xlim(0,NF/10)
ax[1].set_ylabel('smoothed P(ground work)',fontsize=8)
ax[1].tick_params(labelsize=8,labelbottom=(nrow==2))
for sp in ('top','right'): ax[1].spines[sp].set_visible(False)

cols=None
if fur is not None:
    cols=[UNKNOWN]+[EXC_COLOR if z==EXCLUDED else BASE[i] for i,z in enumerate(ZONES)]
    cmap=ListedColormap(cols); norm=BoundaryNorm(np.arange(-1.5,len(ZONES)+0.5),cmap.N)
    ax[2].imshow(fur[None,:],aspect='auto',cmap=cmap,norm=norm,
                 extent=[0,NF/10,0,1],interpolation='nearest')
    ax[2].set_yticks([]); ax[2].set_xlim(0,NF/10)
    ax[2].set_ylabel('furrow',fontsize=8,rotation=0,ha='right',va='center')
    for sp in ax[2].spines.values(): sp.set_visible(False)

axL=ax[-1]
axL.set_xlabel('time (s)',fontsize=8,labelpad=2)
axL.tick_params(labelsize=8,labelbottom=True)

for arr in [~V]:
    prev=arr[0]; st=0
    for i in range(1,NF+1):
        if i==NF or arr[i]!=prev:
            if prev:
                for A in ax: A.axvspan(st/10,i/10,color=C_NOLIDAR,alpha=0.25,lw=0)
            if i<NF: prev=arr[i]; st=i

if axm is not None:
    axm.imshow(mapimg); axm.axis('off')

h=[Patch(color=C_FP,alpha=0.28,label='false positive'),
   Patch(color=C_FN,alpha=0.28,label='false negative'),
   Patch(color=C_NOLIDAR,alpha=0.25,label='no LiDAR data')]
if fur is not None:
    h+=[Patch(facecolor=cols[ZONES.index(z)+1],
              label=z+(' (excluded)' if z==EXCLUDED else '')) for z in present]

fig.text(0.045,0.965,'(a)',fontsize=10,fontweight='bold',va='top')
if axm is not None:
    fig.text(1-MAP_FRAC+0.115,0.965,'(b)',fontsize=10,fontweight='bold',va='top')

# 범례 항목이 모두 (a) 에 관한 것이므로 (a) 폭 안에서 가운데 정렬
_LX = 0.045 + (0.985-0.045)*(1-MAP_FRAC)/2 if axm is not None else 0.5
fig.legend(handles=h,loc='upper center',bbox_to_anchor=(_LX,0.055),
           fontsize=7.5,frameon=False,ncol=len(h),
           handlelength=1.4,columnspacing=1.4,handletextpad=0.5)
fig.subplots_adjust(bottom=0.17)

plt.savefig('results_figures/fig5_timeline.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.savefig('results_figures/fig5_timeline.pdf',bbox_inches='tight',facecolor='white')
print('저장 완료')
print(f'  가동률 {gg[V].mean()*100:.1f}%  카메라 꺼짐 {(1-gg[V].mean())*100:.1f}%')

VIEWER=shutil.which("eog") or shutil.which("xdg-open")
if VIEWER:
    subprocess.Popen([VIEWER,'results_figures/fig5_timeline.png'],
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
