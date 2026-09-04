#!/usr/bin/env python3
"""PointNet 게이트, 네이티브 반환 수 + 패딩 마스크.

gate_pointnet_binary.py 는 20~50개 점을 512 로 복제해 넣었다. 이 버전은
24개로 맞추고 패딩 자리를 풀링에서 제외한다. 평가는 gate_final.py 와
같은 규약(인과 평활, 가동률 고정)을 쓰므로 여기서는 점수만 저장한다."""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F_, os, glob, json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import open3d as o3d

NUM      = 24
EPOCHS   = 20
OUT      = 's8_score_pn_native.npy'
S8_CACHE = 's8_pcd_cache_native.npz'
W_SMOOTH = 24
N_LIDAR  = 2768
TARGET_DUTY = 0.557

d=np.load('pcd_cache_native.npz',allow_pickle=True)
P=d['pts']; M=d['mask']; Y=d['labels']
POS_TR=['homi_work','squatting','squat_moving']
yb=np.isin(Y,POS_TR).astype(np.int64)
print(f'학습 {len(P)}  대상 {yb.sum()} ({yb.mean()*100:.1f}%)  점 {P.shape[1]}')


class PN(nn.Module):
    """패딩 자리를 max-pooling 에서 제외한다"""
    def __init__(s):
        super().__init__()
        s.c1=nn.Conv1d(3,64,1); s.c2=nn.Conv1d(64,128,1); s.c3=nn.Conv1d(128,512,1)
        s.b1=nn.BatchNorm1d(64); s.b2=nn.BatchNorm1d(128); s.b3=nn.BatchNorm1d(512)
        s.f1=nn.Linear(512,256); s.f2=nn.Linear(256,2)
        s.b4=nn.BatchNorm1d(256); s.d=nn.Dropout(0.3)
    def forward(s,x,m):
        x=x.transpose(2,1)
        x=F_.relu(s.b1(s.c1(x))); x=F_.relu(s.b2(s.c2(x))); x=F_.relu(s.b3(s.c3(x)))
        x=x.masked_fill(~m.unsqueeze(1), -1e4)     # 패딩 제외
        x=x.max(2)[0]
        x=s.d(F_.relu(s.b4(s.f1(x))))
        return s.f2(x)


dev='cuda' if torch.cuda.is_available() else 'cpu'
w=torch.tensor([1.0/max((yb==0).sum(),1),1.0/max((yb==1).sum(),1)],dtype=torch.float32)
w=(w/w.sum()*2).to(dev)
net=PN().to(dev)
opt=torch.optim.Adam(net.parameters(),1e-3); lo=nn.CrossEntropyLoss(weight=w)
dl=DataLoader(TensorDataset(torch.tensor(P),torch.tensor(M),torch.tensor(yb)),
              batch_size=48,shuffle=True)
print(f'device {dev}, 학습 시작')
for ep in range(EPOCHS):
    net.train(); tot=0
    for xb,mb,ybb in dl:
        xb,mb,ybb=xb.to(dev),mb.to(dev),ybb.to(dev)
        opt.zero_grad(); l=lo(net(xb,mb),ybb); l.backward(); opt.step(); tot+=l.item()
    if ep%5==0: print(f'  ep{ep} loss {tot/len(dl):.4f}')

# ── S8 캐시 ──
if os.path.exists(S8_CACHE):
    c=np.load(S8_CACHE); Ps,Ms,Fs=c['pts'],c['mask'],c['frames']
    print(f'{S8_CACHE} 재사용  {Ps.shape}')
else:
    FX=(1.34,4.14); FY=(-1.99,3.01)
    bg=np.load('background_model.npy'); tree=KDTree(bg)
    ROI={'x':(-1.2,3.85),'y':(-2.0,3.3),'z':(-2.0,1.5)}
    fs=sorted(glob.glob('pcd/S8/*.pcd')); np.random.seed(42)
    Pl=[];Ml=[];Fl=[]
    print('S8 캐시 생성...')
    for i,f in enumerate(fs):
        if i%500==0: print(f'  {i}/{len(fs)}')
        p=np.asarray(o3d.io.read_point_cloud(f).points)
        m=((p[:,0]>ROI['x'][0])&(p[:,0]<ROI['x'][1])&(p[:,1]>ROI['y'][0])&
           (p[:,1]<ROI['y'][1])&(p[:,2]>ROI['z'][0])&(p[:,2]<ROI['z'][1]))
        p=p[m]
        if len(p)<10: continue
        dd,_=tree.query(p,k=1); per=p[dd>0.05]
        if len(per)<12: continue
        lb=DBSCAN(eps=0.30,min_samples=3).fit_predict(per)
        best=None;bn=0
        for c_ in set(lb):
            if c_<0: continue
            q=per[lb==c_]
            if len(q)<10 or np.ptp(q[:,2])<0.25: continue
            cx,cy=q[:,0].mean(),q[:,1].mean()
            if not(FX[0]<=cx<=FX[1] and FY[0]<=cy<=FY[1]): continue
            if len(q)>bn: best,bn=q,len(q)
        if best is None: continue
        n=len(best); q=np.zeros((NUM,3),np.float32); msk=np.zeros(NUM,bool)
        if n>=NUM:
            v=best[np.random.choice(n,NUM,replace=False)]; msk[:]=True
        else:
            v=best; msk[:n]=True
        v=v.astype(np.float32); v=v-v.mean(0)
        sc_=np.max(np.linalg.norm(v,axis=1))
        if sc_>0: v=v/sc_
        q[:len(v)]=v
        Pl.append(q); Ml.append(msk); Fl.append(i)
    Ps=np.array(Pl); Ms=np.array(Ml); Fs=np.array(Fl)
    np.savez(S8_CACHE, pts=Ps, mask=Ms, frames=Fs)
    print(f'S8 {len(Ps)}프레임 저장')

net.eval(); pb=[]
with torch.no_grad():
    for i in range(0,len(Ps),64):
        pb.append(torch.softmax(net(torch.tensor(Ps[i:i+64]).to(dev),
                                    torch.tensor(Ms[i:i+64]).to(dev)),1)[:,1].cpu().numpy())
pb=np.concatenate(pb)

a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True); NF=len(gt)
POS=json.load(open('label_standard.json'))['ground_work_gt']
t=np.isin(gt,POS)
li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LIDAR)
pm={int(Fs[i]):float(pb[i]) for i in range(len(Fs))}
s8=np.array([pm.get(int(li[c]),0.0) for c in range(NF)])
np.save(OUT,s8)

# ── 논문과 같은 규약으로 평가 ──
def causal_mean(x,w_):
    return np.convolve(np.concatenate([np.full(w_-1,x[0]),x]),np.ones(w_)/w_,mode='valid')

def PRF(g):
    tp=int((g&t&V).sum()); fp=int((g&~t&V).sum()); fn=int((~g&t&V).sum())
    P_=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    return P_*100,R*100,2*P_*R/max(P_+R,1e-9)*100,g[V].mean()*100

print(f'\nAUC (평활 전) {roc_auc_score(t[V],s8[V]):.3f}')
for nm,f in (('native (24, masked)',OUT),('512 replicated',
                                          's8_score_pn_bin.npy')):
    if not os.path.exists(f): continue
    s=causal_mean(np.load(f)[:NF],W_SMOOTH)
    th=min(np.arange(0.01,1.00,0.005),key=lambda x:abs((s[V]>x).mean()-TARGET_DUTY))
    P_,R,F,d=PRF(s>th)
    print(f'  {nm:22s} Thr {th:.2f}  P {P_:5.1f}  R {R:5.1f}  F1 {F:5.1f}  duty {d:5.1f}')
