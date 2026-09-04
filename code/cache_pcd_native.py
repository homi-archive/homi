#!/usr/bin/env python3
"""PointNet 학습 캐시: 실제 반환 수에 맞춘 목표 크기 + 패딩 마스크

기존 cache_pcd_v4.py 는 20~50개 점을 512 로 복제해 채웠다. 이 버전은
NUM 개로 맞추되 부족분을 복제 대신 0 으로 패딩하고 마스크를 함께
저장해, 풀링에서 실제 점만 반영되게 한다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, open3d as o3d, glob
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

NUM = 24                        # 목표 점 수 (전체 분포 중앙값 23)
OUT = 'pcd_cache_native.npz'
FX=(1.34,4.14); FY=(-1.99,3.01)
ROI={'x':(-1.2,3.85),'y':(-2.0,3.3),'z':(-2.0,1.5)}
SC={'S2':'standing','S3':'walking','S4':'squatting',
    'S5':'squat_moving','S6':'watering','S7':'homi_work'}

bg=np.load('background_model.npy'); tree=KDTree(bg)
np.random.seed(42)


def cluster(f):
    p=np.asarray(o3d.io.read_point_cloud(f).points)
    m=((p[:,0]>ROI['x'][0])&(p[:,0]<ROI['x'][1])&
       (p[:,1]>ROI['y'][0])&(p[:,1]<ROI['y'][1])&
       (p[:,2]>ROI['z'][0])&(p[:,2]<ROI['z'][1]))
    p=p[m]
    if len(p)<10: return None
    dd,_=tree.query(p,k=1); per=p[dd>0.05]
    if len(per)<12: return None
    lb=DBSCAN(eps=0.30,min_samples=3).fit_predict(per)
    best=None;bn=0
    for c in set(lb):
        if c<0: continue
        q=per[lb==c]
        if len(q)<10 or np.ptp(q[:,2])<0.25: continue
        cx,cy=q[:,0].mean(),q[:,1].mean()
        if not(FX[0]<=cx<=FX[1] and FY[0]<=cy<=FY[1]): continue
        if len(q)>bn: best,bn=q,len(q)
    return best


def pack(best):
    """NUM 개로 맞춤. 많으면 무작위 추출, 적으면 0 패딩 + 마스크"""
    n=len(best)
    q=np.zeros((NUM,3),np.float32); msk=np.zeros(NUM,bool)
    if n>=NUM:
        idx=np.random.choice(n,NUM,replace=False)
        v=best[idx]; msk[:]=True
    else:
        v=best; msk[:n]=True
    v=v.astype(np.float32); v=v-v.mean(0)
    sc=np.max(np.linalg.norm(v,axis=1))
    if sc>0: v=v/sc
    q[:len(v)]=v
    return q,msk,n


P=[];M=[];N=[];Y=[];S_=[]
for S,lab in SC.items():
    fs=sorted(glob.glob(f'pcd/{S}/*.pcd'))
    print(f'{S} {len(fs)}...')
    for f in fs:
        best=cluster(f)
        if best is None: continue
        q,msk,n=pack(best)
        P.append(q); M.append(msk); N.append(n); Y.append(lab); S_.append(S)

P=np.array(P); M=np.array(M); N=np.array(N)
np.savez('pcd_cache_native.npz', pts=P, mask=M, npts=N,
         labels=np.array(Y), sessions=np.array(S_))
print(f'\n저장: {OUT}  {P.shape}')
print(f'  실제 점 수 중앙값 {int(np.median(N))}  5~95% {np.percentile(N,[5,95]).round(0)}')
print(f'  {NUM} 이상인 프레임 {int((N>=NUM).sum())}/{len(N)} = {(N>=NUM).mean()*100:.1f}%')
print(f'  평균 유효 비율 {M.mean()*100:.1f}%')
