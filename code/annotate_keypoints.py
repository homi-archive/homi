#!/usr/bin/env python3
"""S8 호미질 구간의 팔 관절 수동 주석. PCK 계산용 정답을 만든다.

모델 예측은 화면에 띄우지 않는다. 예측을 보고 찍으면 PCK 가 부풀려진다.

  마우스 좌클릭   현재 관절 위치를 찍고 다음 관절로
  s               현재 관절 건너뛰기 (보이지 않을 때)
  u               직전 클릭 취소
  r               이 프레임 처음부터 다시
  n               이 프레임을 통째로 건너뛰기
  +/-             확대/축소 (기본 1.5배)
  q               저장하고 종료

여섯 관절을 다 찍으면 자동으로 다음 프레임으로 넘어간다.
"""
import numpy as np, cv2, glob, json, os

N_SAMPLE = 200
OUT      = 'keypoint_gt_s8.json'
# left/right 는 카메라가 아니라 피험자 기준 (COCO 규약, YOLOv8 과 동일)
JOINTS   = ["subject's LEFT shoulder","subject's RIGHT shoulder",
            "subject's LEFT elbow","subject's RIGHT elbow",
            "subject's LEFT wrist","subject's RIGHT wrist"]
COL      = [(90,200,90),(90,200,90),(255,180,60),
            (255,180,60),(90,120,255),(90,120,255)]

gt=np.load('s8_groundtruth_v3.npy',allow_pickle=True)
cam=sorted(glob.glob('video_S8_frames/*.jpg'))
homi=np.flatnonzero(gt=='homi_work')
sel=sorted(homi[np.linspace(0,len(homi)-1,N_SAMPLE).astype(int)].tolist())
print(f'호미질 {len(homi)}프레임에서 {len(sel)}개 균등 추출')

done={}
if os.path.exists(OUT):
    done={int(k):v for k,v in json.load(open(OUT)).items()}
    print(f'{OUT} 이어서 작업. 완료 {len(done)}프레임')

todo=[f for f in sel if f not in done]
if not todo:
    print('모두 완료'); raise SystemExit

state={'pts':[], 'zoom':1.5, 'click':None}
def on_mouse(ev,x,y,flags,param):
    if ev==cv2.EVENT_LBUTTONDOWN: state['click']=(x,y)

cv2.namedWindow('keypoints')
cv2.setMouseCallback('keypoints', on_mouse)
print(__doc__)

i=0
while i < len(todo):
    f=todo[i]
    img=cv2.imread(cam[f])
    while True:
        z=state['zoom']
        d=cv2.resize(img,None,fx=z,fy=z)
        k=len(state['pts'])
        # 찍은 점 표시
        for j,p in enumerate(state['pts']):
            if p is None: continue
            cv2.circle(d,(int(p[0]*z),int(p[1]*z)),6,COL[j],-1)
            cv2.circle(d,(int(p[0]*z),int(p[1]*z)),8,(255,255,255),1)
        # 헤더
        hdr=np.full((104,d.shape[1],3),255,np.uint8)
        cv2.putText(hdr,f'CAM {f}   [{i+1}/{len(todo)}]  done {len(done)}',
                    (12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(60,60,60),1,cv2.LINE_AA)
        if k<6:
            cv2.putText(hdr,f'>>> CLICK: {JOINTS[k]}',(12,76),
                        cv2.FONT_HERSHEY_SIMPLEX,.95,COL[k],2,cv2.LINE_AA)
        else:
            cv2.putText(hdr,'6/6 done, saving...',(12,76),
                        cv2.FONT_HERSHEY_SIMPLEX,.9,(40,160,40),2,cv2.LINE_AA)
        # 남은 관절 목록
        x=12
        for j,nm in enumerate(JOINTS):
            mark='o' if j<k and state['pts'][j] is not None else \
                 ('x' if j<k else '.')
            c=COL[j] if j<k else (170,170,170)
            t=f'{mark} {nm}'
            cv2.putText(hdr,t,(x,94),cv2.FONT_HERSHEY_SIMPLEX,.42,c,1,cv2.LINE_AA)
            x+=int(cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,.42,1)[0][0])+18
        cv2.putText(hdr,'LEFT/RIGHT = the worker\'s own side, not the screen side',
                    (12,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,200),1,cv2.LINE_AA)
        cv2.putText(hdr,'s skip joint   u undo   r restart   n skip frame   +/- zoom   q quit',
                    (d.shape[1]-660 if d.shape[1]>700 else 12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,(140,140,140),1,cv2.LINE_AA)
        cv2.imshow('keypoints', np.vstack([hdr,d]))

        if k>=6:
            done[f]=[list(p) if p else None for p in state['pts']]
            json.dump({str(a):b for a,b in done.items()},open(OUT,'w'))
            state['pts']=[]; i+=1; break

        key=cv2.waitKey(20)&0xFF
        if state['click'] is not None:
            cx,cy=state['click']; state['click']=None
            if cy>104:
                state['pts'].append(((cx)/z,(cy-104)/z))
        elif key==ord('s'): state['pts'].append(None)
        elif key==ord('u'):
            if state['pts']: state['pts'].pop()
        elif key==ord('r'): state['pts']=[]
        elif key==ord('n'): state['pts']=[]; i+=1; break
        elif key in (ord('+'),ord('=')): state['zoom']=min(z+0.25,3.0)
        elif key==ord('-'): state['zoom']=max(z-0.25,0.5)
        elif key==ord('q'):
            json.dump({str(a):b for a,b in done.items()},open(OUT,'w'))
            cv2.destroyAllWindows()
            print(f'\n저장: {OUT}  {len(done)}프레임')
            raise SystemExit

cv2.destroyAllWindows()
json.dump({str(a):b for a,b in done.items()},open(OUT,'w'))
print(f'\n완료: {OUT}  {len(done)}프레임')
