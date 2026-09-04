#!/usr/bin/env python3
"""반사강도 두 모집단: 인물 클러스터와 밭 전경 전체.
   논문의 6.6/7.3 은 전자(강도<200), 16.9/64.6 은 후자(강도<255)."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import sqlite3, glob, numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

bg=np.load('background_model.npy'); tree=KDTree(bg)
ROI={'x':(-1.2,3.85),'y':(-2.0,3.3),'z':(-2.0,1.5)}
FX=(1.34,4.14); FY=(-1.99,3.01)
SC={'S1_empty':'empty field','S2_idle':'standing',
    'S6_overhead':'spraying','S7_rhythmic':'Ho-Mi work'}

def frames(scene, step=20, limit=40):
    db=glob.glob(f'*/{scene}*_0.db3')
    if not db: return
    con=sqlite3.connect(db[0])
    tid,ttype=list(con.execute(
      "SELECT id,type FROM topics WHERE type LIKE '%PointCloud2%'"))[0]
    rows=list(con.execute(f"SELECT data FROM messages WHERE topic_id={tid}"))
    for k in range(0,len(rows),step):
        if limit<=0: break
        msg=deserialize_message(rows[k][0], get_message(ttype))
        b=np.frombuffer(msg.data,dtype=np.uint8).reshape(-1,msg.point_step)
        xyz=b[:,0:12].copy().view(np.float32).reshape(-1,3)
        it =b[:,16:20].copy().view(np.float32).ravel()
        limit-=1
        yield xyz,it

def split(xyz,it):
    """(밭 전경 전체 강도, 인물 클러스터 강도)"""
    m=((xyz[:,0]>ROI['x'][0])&(xyz[:,0]<ROI['x'][1])&
       (xyz[:,1]>ROI['y'][0])&(xyz[:,1]<ROI['y'][1])&
       (xyz[:,2]>ROI['z'][0])&(xyz[:,2]<ROI['z'][1]))
    p,i2=xyz[m],it[m]
    if len(p)<10: return None,None
    dd,_=tree.query(p,k=1); keep=dd>0.05
    per,ip=p[keep],i2[keep]
    if len(per)<12: return None,None
    lb=DBSCAN(eps=0.30,min_samples=3).fit_predict(per)
    best=None;bn=0
    for c in set(lb):
        if c<0: continue
        s=(lb==c); q=per[s]
        if len(q)<10 or np.ptp(q[:,2])<0.25: continue
        cx,cy=q[:,0].mean(),q[:,1].mean()
        if not(FX[0]<=cx<=FX[1] and FY[0]<=cy<=FY[1]): continue
        if len(q)>bn: best,bn=ip[s],len(q)
    return ip, best

print(f'{"scenario":14s}{"fg n":>8s}{"fg<255":>9s}{"clu n":>8s}{"clu<200":>9s}{"clu<255":>9s}')
out={}
for s,nm in SC.items():
    FG=[];CL=[]
    for xyz,it in frames(s):
        f,c=split(xyz,it)
        if f is not None: FG.append(f)
        if c is not None: CL.append(c)
    if not FG: print(f'{nm:14s} 데이터 없음'); continue
    F=np.concatenate(FG)
    C=np.concatenate(CL) if CL else np.array([])
    print(f'{nm:14s}{len(F):8d}{(F<255).mean()*100:8.1f}%{len(C):8d}'
          f'{((C<200).mean()*100 if len(C) else float("nan")):8.1f}%'
          f'{((C<255).mean()*100 if len(C) else float("nan")):8.1f}%')
    out[nm]={'fg':F,'cluster':C}
np.save('intensity_two_populations.npy', out, allow_pickle=True)
print('\n저장: intensity_two_populations.npy')
