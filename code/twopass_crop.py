#!/usr/bin/env python3
"""색 마스크 없는 크롭. 전체 프레임에서 사람을 한 번 찾고 그 상자를
   잘라 다시 자세를 추정한다. 참가자별 휴리스틱이 필요 없다."""
import numpy as np, json, glob, cv2
from ultralytics import YOLO

CONF, PAD = 0.3, 0.60          # 상자 여유 60%
ARM = [5,6,7,8,9,10]
frames = sorted(glob.glob('video_S7_frames/*.jpg'))
m = YOLO('yolov8n-pose.pt')

ok=det=0
for i,fp in enumerate(frames):
    if i%300==0: print(f'  {i}/{len(frames)}')
    im=cv2.imread(fp); H,W=im.shape[:2]
    r=m(im, verbose=False, conf=0.25)[0]
    if r.boxes is None or len(r.boxes)==0: continue
    det+=1
    a=(r.boxes.xyxy[:,2]-r.boxes.xyxy[:,0])*(r.boxes.xyxy[:,3]-r.boxes.xyxy[:,1])
    x1,y1,x2,y2=r.boxes.xyxy[int(a.argmax())].cpu().numpy()
    w,h=x2-x1,y2-y1
    x1=max(0,int(x1-w*PAD)); y1=max(0,int(y1-h*PAD))
    x2=min(W,int(x2+w*PAD)); y2=min(H,int(y2+h*PAD))
    if x2-x1<20 or y2-y1<20: continue
    r2=m(im[y1:y2,x1:x2], verbose=False, conf=0.25)[0]
    if r2.keypoints is None or len(r2.keypoints.data)==0: continue
    j=0
    if r2.boxes is not None and len(r2.boxes)>1:
        a2=(r2.boxes.xyxy[:,2]-r2.boxes.xyxy[:,0])*(r2.boxes.xyxy[:,3]-r2.boxes.xyxy[:,1])
        j=int(a2.argmax())
    k=r2.keypoints.data[j].cpu().numpy()
    if all(k[q,2]>CONF for q in ARM): ok+=1

n=len(frames)
print(f'\n2단계 크롭 (색 무관)  사람 검출 {det}/{n} = {det/n*100:.1f}%'
      f'   팔 6점 {ok}/{n} = {ok/n*100:.1f}%')
print(f'  비교: 전체 프레임 58.0%,  HSV 크롭 74.6%')
json.dump({'n':n,'person':det,'arm6':ok,'rate':ok/n},
          open('twopass_crop_pad60.json','w'), indent=1)
