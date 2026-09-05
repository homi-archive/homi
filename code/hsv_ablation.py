#!/usr/bin/env python3
"""HSV 크롭이 자세 검출에 얼마나 기여하는가.
   같은 S7 프레임에 (a) 크롭 없이 전체 프레임, (b) HSV 크롭을 각각 돌려 비교한다."""
import numpy as np, json, glob, cv2
from ultralytics import YOLO

CONF = 0.3
ARM  = [5,6,7,8,9,10]      # COCO: 양쪽 어깨, 팔꿈치, 손목
frames = sorted(glob.glob('video_S7_frames/*.jpg'))
crop   = sorted(glob.glob('video_S7_frames_crop/*.jpg'))
print(f'원본 {len(frames)}  크롭 {len(crop)}')

m = YOLO('yolov8n-pose.pt')

def run(paths, tag):
    ok = 0; det = 0
    for i in range(0, len(paths), 1):
        r = m(paths[i], verbose=False, conf=0.25)[0]
        if r.keypoints is None or len(r.keypoints.data) == 0: continue
        det += 1
        # 사람이 여럿이면 가장 큰 상자를 고른다 (크롭 없을 때의 자연스러운 규칙)
        if r.boxes is not None and len(r.boxes) > 1:
            a = (r.boxes.xyxy[:,2]-r.boxes.xyxy[:,0])*(r.boxes.xyxy[:,3]-r.boxes.xyxy[:,1])
            j = int(a.argmax())
        else:
            j = 0
        k = r.keypoints.data[j].cpu().numpy()
        if all(k[q,2] > CONF for q in ARM): ok += 1
        if i % 300 == 0: print(f'  {tag} {i}/{len(paths)}')
    print(f'{tag:14s} 사람 검출 {det}/{len(paths)} = {det/len(paths)*100:.1f}%'
          f'   팔 6점 conf>{CONF} {ok}/{len(paths)} = {ok/len(paths)*100:.1f}%')
    return ok, len(paths)

a = run(frames, 'full frame')
b = run(crop,   'HSV crop')
json.dump({'full_frame':a, 'hsv_crop':b}, open('hsv_ablation.json','w'), indent=1)
print('\n저장: hsv_ablation.json')
