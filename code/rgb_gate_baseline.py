#!/usr/bin/env python3
"""RGB 프레임 차분 게이트 기준선.

서론은 픽셀 변화 트리거가 바람에 흔들리는 식생에 취약하다고 논증한다.
여기서는 같은 영상으로 그 트리거를 실제로 만들어 같은 가동률에서 비교한다.
프레임을 저해상도로 낮춰 저전력 상시 비전 노드를 흉내낸다."""
import numpy as np, cv2, glob, json, os

W_SM, DUTY, N_LID = 24, 0.557, 2768
SIZES = [(32,18), (64,36), (160,90)]

cam=sorted(glob.glob('video_S8_frames/*.jpg'))
gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
POS=json.load(open('label_standard.json'))['ground_work_gt']
a,b=np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') else np.load('sync_model.npy')
NF=len(gt); li=np.array([int(round(a*c+b)) for c in range(NF)])
V=(li>=0)&(li<N_LID); t=np.isin(gt[:NF],POS)

def cm(x,w): return np.convolve(np.concatenate([np.full(w-1,x[0]),x]),np.ones(w)/w,mode='valid')
def PRF(g):
    tp=int((g&t&V).sum()); fp=int((g&~t&V).sum()); fn=int((~g&t&V).sum())
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    return P*100,R*100,2*P*R/max(P+R,1e-9)*100,g[V].mean()*100

print(f'{"resolution":14s}{"smoothing":11s}{"P":>7s}{"R":>7s}{"F1":>7s}{"duty":>7s}')
out={}
for w,h in SIZES:
    prev=None; d=np.zeros(NF)
    for i in range(NF):
        im=cv2.cvtColor(cv2.resize(cv2.imread(cam[i]),(w,h)),cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: d[i]=np.abs(im-prev).mean()
        prev=im
    for sm in (False,True):
        s=cm(d,W_SM) if sm else d
        th=min(np.percentile(s[V],np.arange(1,100,0.5)),
               key=lambda x:abs((s[V]>x).mean()-DUTY))
        P,R,F,du=PRF(s>th)
        print(f'{f"{w}x{h}":14s}{"2 s" if sm else "none":11s}'
              f'{P:7.1f}{R:7.1f}{F:7.1f}{du:7.1f}')
        out[f'{w}x{h}_{"sm" if sm else "raw"}']=[P,R,F,du]
json.dump(out,open('rgb_gate_baseline.json','w'),indent=1)
print('\n저장: rgb_gate_baseline.json')
