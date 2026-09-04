#!/usr/bin/env python3
"""상단 뷰 맵 v2: 히트맵 누적 수정, F6 축소, 유령 억제"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2, glob, os, json, collections, open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# ══════ 조정 값 ══════
X_SHIFT  = 1.4
X_FAR    = 4.15
X_F6     = 4.28
F6_W     = 0.22          # F6 폭
LEN      = {'R1':3.48,'R2':3.48,'R3':2.03,'R4':2.03,'R5':2.97,'R6':2.97}
SMOOTH   = 5
MAX_STEP = 0.09
JUMP_REJ = 0.35          # 이보다 크게 튀면 그 프레임 버림 (유령 억제)
TRAIL_A  = 0.004         # 동선 점 강도 (낮을수록 천천히 진해짐)
MINPTS   = 22
# ════════════════════

ROI={'x':(-1.2,6.0),'y':(-2.6,3.6),'z':(-2.0,1.5)}
bg=np.load('background_model.npy'); tree=KDTree(bg)
fs=sorted(glob.glob('pcd/S8/*.pcd'))
gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
gate=np.load('s8_gate_final_v2.npy')
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
LAY=json.load(open('plot_layout.json'))
RG=sorted(LAY['ridges'],key=lambda r:r['c'])
STD=json.load(open('label_standard.json')); LBL=STD['paper_labels']
cam=sorted(glob.glob('video_S8_frames/*.jpg'))
TANK=np.array([-0.24,1.71])
for i,r in enumerate(RG):
    r['name']=f'R{i+1}'; r['x0']=X_FAR-LEN[f'R{i+1}']; r['x1']=X_FAR

FU=[dict(name=f'F{k+1}',ylo=RG[k]['hi'],yhi=RG[k+1]['lo'],
         xlo=min(RG[k]['x0'],RG[k+1]['x0']),xhi=X_F6+F6_W/2) for k in range(5)]
# 고랑 폭 조정: F3, F5는 좁고 남는 폭은 양옆 두둑이 흡수
NARROW={'F3':0.55,'F5':0.60}
for k,f in enumerate(FU):
    if f['name'] not in NARROW: continue
    lo0,hi0=f['ylo'],f['yhi']
    c=(lo0+hi0)/2; hw=(hi0-lo0)/2*NARROW[f['name']]
    f['ylo'],f['yhi']=c-hw,c+hw
    RG[k]['hi']   = f['ylo']          # 아래쪽 두둑이 확장
    RG[k+1]['lo'] = f['yhi']          # 위쪽 두둑이 확장
    RG[k]['c']    = (RG[k]['lo']+RG[k]['hi'])/2
    RG[k+1]['c']  = (RG[k+1]['lo']+RG[k+1]['hi'])/2
    print(f"  {f['name']}  {hi0-lo0:.3f} -> {2*hw:.3f}m   "
          f"{RG[k]['name']} {RG[k]['hi']-RG[k]['lo']:.3f}m  "
          f"{RG[k+1]['name']} {RG[k+1]['hi']-RG[k+1]['lo']:.3f}m")

YLO=min(r['lo'] for r in RG); YHI=max(r['hi'] for r in RG)
F6=dict(name='F6',ylo=YLO,yhi=YHI,xlo=X_FAR,xhi=X_F6+F6_W/2)

def _soft(v,lo,hi):
    """[lo,hi] 안으로 부드럽게 압축. 경계에 붙지 않고 편차가 남는다."""
    c=(lo+hi)/2; hw=(hi-lo)/2
    if hw<=1e-6: return c
    return c+hw*np.tanh((v-c)/(hw*2.6))

LAST_SEG=[None]
def confine(p):
    best=None
    for f in FU+[F6]:
        if f['name']=='F6' and p[0]<X_FAR-0.10: continue
        q=np.array([np.clip(p[0],f['xlo'],f['xhi']),
                    np.clip(p[1],f['ylo'],f['yhi'])])
        d=np.linalg.norm(p-q)
        if best is None or d<best[0]: best=(d,f)
    f = best[1] if best is not None else F6
    if LAST_SEG[0] is not None and f['name']!=LAST_SEG[0]:
        for g in FU+[F6]:
            if g['name']!=LAST_SEG[0]: continue
            q=np.array([np.clip(p[0],g['xlo'],g['xhi']),
                        np.clip(p[1],g['ylo'],g['yhi'])])
            if np.linalg.norm(p-q) < best[0]+0.22:   # 관성 여유
                f=g
            break
    LAST_SEG[0]=f['name']
    # 통로 긴 방향은 자유롭게, 좁은 방향만 부드럽게 압축
    if f['name']=='F6':
        q=np.array([_soft(p[0],f['xlo'],f['xhi']),
                    np.clip(p[1],f['ylo'],f['yhi'])])
    else:
        q=np.array([np.clip(p[0],f['xlo'],f['xhi']),
                    _soft(p[1],f['ylo'],f['yhi'])])
    return q,f['name']

CACHE={}
def person(i):
    if i in CACHE: return CACHE[i]
    p=np.asarray(o3d.io.read_point_cloud(fs[i]).points)
    m=((p[:,0]>ROI['x'][0])&(p[:,0]<ROI['x'][1])&
       (p[:,1]>ROI['y'][0])&(p[:,1]<ROI['y'][1])&
       (p[:,2]>ROI['z'][0])&(p[:,2]<ROI['z'][1]))
    p=p[m]; out=None
    if len(p)>=10:
        dd,_=tree.query(p,k=1); per=p[dd>0.05]
        if len(per)>=12:
            lb=DBSCAN(eps=0.30,min_samples=3).fit_predict(per); bn=0
            for c in set(lb):
                if c<0: continue
                q=per[lb==c]
                if len(q)<MINPTS or np.ptp(q[:,2])<0.25: continue
                if len(q)>bn: out,bn=q,len(q)
    CACHE[i]=out; return out

C0,C1=106,2412
print('1단계 원 위치 + 점프 제거')
raw={}; last=None
for ci in range(C0,C1+1):
    li=int(round(a*ci+b))
    q=person(li) if 0<=li<len(fs) else None
    v=None if q is None else q[:,:2].mean(0)+np.array([X_SHIFT,0.0])
    if v is not None and last is not None and np.linalg.norm(v-last)>JUMP_REJ:
        v=None                       # 유령: 버림
    if v is not None: last=v
    raw[ci]=v
print(f'  유효 {sum(1 for v in raw.values() if v is not None)}/{C1-C0+1}')

print('2단계 평활 + 구속')
pos={}; cur=None
for ci in range(C0,C1+1):
    w=[raw[k] for k in range(max(C0,ci-SMOOTH//2),min(C1,ci+SMOOTH//2)+1)
       if raw.get(k) is not None]
    if not w:
        pos[ci]=None if cur is None else (cur.copy(),confine(cur)[1]); continue
    tgt,_=confine(np.median(np.array(w),axis=0))
    if cur is None: cur=tgt
    else:
        d=tgt-cur; n=np.linalg.norm(d)
        cur = tgt if n<=MAX_STEP else cur+d/n*MAX_STEP
    cur,nm=confine(cur)
    pos[ci]=(cur.copy(),nm)
print('  통로별:',dict(collections.Counter(p[1] for p in pos.values() if p)))

XR=(0.30,4.80); YR=(-2.30,3.30)
MW,MH,M=830,640,46
def T(x,y):
    return (int(M+(YR[1]-y)/(YR[1]-YR[0])*(MW-2*M)),
            int(M+(XR[1]-x)/(XR[1]-XR[0])*(MH-2*M)))

BASE=np.full((MH,MW,3),255,np.uint8)
for f in FU+[F6]:
    cv2.rectangle(BASE,T(f['xhi'],f['yhi']),T(f['xlo'],f['ylo']),(245,245,245),-1)
for r in RG:
    cv2.rectangle(BASE,T(r['x1'],r['hi']),T(r['x0'],r['lo']),(222,222,222),-1)
    m=T((r['x0']+r['x1'])/2,(r['lo']+r['hi'])/2)   # 도형 정중앙
    (tw,th),_=cv2.getTextSize(r['name'],cv2.FONT_HERSHEY_SIMPLEX,0.52,2)
    cv2.putText(BASE,r['name'],(m[0]-tw//2,m[1]+th//2),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,(140,140,140),2,cv2.LINE_AA)
for f in FU:
    m=T((f['xlo']+f['xhi'])/2,(f['ylo']+f['yhi'])/2)   # 통로 정중앙
    (tw,th),_=cv2.getTextSize(f['name'],cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
    cv2.putText(BASE,f['name'],(m[0]-tw//2,m[1]+th//2),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,40),2,cv2.LINE_AA)
m=T((F6['xlo']+F6['xhi'])/2,(YLO+YHI)/2)
(tw,th),_=cv2.getTextSize('F6',cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
cv2.putText(BASE,'F6',(m[0]-tw//2,m[1]+th//2),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,40),2,cv2.LINE_AA)
cv2.circle(BASE,T(*TANK),15,(240,240,240),-1)
cv2.putText(BASE,'tank',(T(*TANK)[0]-14,T(*TANK)[1]+27),
            cv2.FONT_HERSHEY_SIMPLEX,0.36,(160,160,160),1,cv2.LINE_AA)
LIDAR_MY=RG[1]['hi']      # R2의 화면상 왼쪽 경계
cv2.circle(BASE,T(0.36,LIDAR_MY),6,(80,80,80),-1)
cv2.putText(BASE,'LiDAR',(T(0.36,LIDAR_MY)[0]-18,T(0.36,LIDAR_MY)[1]+22),
            cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,80,80),1,cv2.LINE_AA)

ACC=np.zeros((MH,MW),np.float32)          # 방문 누적 (스칼라)
BLUE_LO=np.array([245,205,180],np.float32)   # 연한 파랑 BGR
BLUE_HI=np.array([200, 40, 20],np.float32)   # 진한 파랑

CW,CH=960,540; PAD=max(CH,MH)
out=cv2.VideoWriter('results_figures/field_map_w6.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'),10.0,(CW+MW,PAD+44))
print('3단계 렌더')
for ci in range(C0,C1+1):
    if (ci-C0)%300==0: print(f'  {ci-C0}/{C1-C0}')
    st=pos.get(ci); txt='no detection'
    if st is not None:
        p,nm=st
        lay=np.zeros((MH,MW),np.float32)
        cv2.circle(lay,T(*p),25,1.0,-1,cv2.LINE_AA)
        ACC=np.clip(ACC+lay*TRAIL_A,0,1.0)
        txt=f'{nm}   x {p[0]:.2f}  y {p[1]:+.2f}'
    mp=BASE.astype(np.float32).copy()
    t=ACC[...,None]
    col=BLUE_LO*(1-t)+BLUE_HI*t          # 강할수록 진한 파랑
    mp=mp*(1-t)+col*t
    mp=mp.astype(np.uint8)
    if st is not None:
        cv2.circle(mp,T(*st[0]),27,(215,160,190),-1)   # 연보라
        cv2.circle(mp,T(*st[0]),27,(120,60,110),2)
    c=cv2.resize(cv2.imread(cam[ci]),(CW,CH))
    left=np.full((PAD,CW,3),255,np.uint8); left[:CH]=c
    right=np.full((PAD,MW,3),255,np.uint8); right[:MH]=mp
    bar=np.full((44,CW+MW,3),255,np.uint8)
    cv2.putText(bar,f'CAM {ci}  t={ci/10:.1f}s  {LBL.get(gt[ci],gt[ci])}  '
                f'gate {"ON" if gate[ci] else "off"}',(14,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(30,30,30),1,cv2.LINE_AA)
    cv2.putText(bar,txt,(CW+14,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                (60,60,60),1,cv2.LINE_AA)
    out.write(np.vstack([np.hstack([left,right]),bar]))
out.release()
print('완료: results_figures/field_map_w6.mp4')
