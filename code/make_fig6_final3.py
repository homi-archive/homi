#!/usr/bin/env python3
"""Fig 6: 위 사람 / 아래 로봇, 같은 4위상"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2, json, mujoco, glob, os, subprocess, shutil

# ── 파라미터 ──
PHASE=[('start',  0,153),
       ('lift',   6,155),
       ('top',    9,162),
       ('pull',  16,165)]
HUMAN_SKEL = True     # 사람 패널에 스켈레톤 표시
BG_TOL     = 40       # 배경 제거 임계값. 90 은 로봇 밝은 면까지 지웠음
OUT        = ('results_figures/fig6_strokes.png' if HUMAN_SKEL
              else 'results_figures/fig6_strokes_noskel.png')

src=open('render_pipeline_final2.py').read()
exec(src[:src.index('cam_all=')], globals())
ABD_GAIN=1.0; FLEX_GAIN=1.0
cam.distance,cam.elevation=1.75,-16
cam.lookat=np.array([0.20,0.0,GROUND+0.32])

S=json.load(open('homi_two_strokes.json'))
J=S['joints']; NS=S['profile_steps']; PN=S['strokes']['normal']
print('위상:',PHASE,'  스켈레톤:',HUMAN_SKEL,'  BG_TOL:',BG_TOL)

SK=json.load(open('yolo_skeleton_s7.json'))
cams=sorted(glob.glob('video_S7_frames_crop/*.jpg'))
LINK=[('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
      ('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
      ('left_shoulder','right_shoulder')]
KP=['right_shoulder','right_elbow','right_wrist',
    'left_shoulder','left_elbow','left_wrist']

def human(idx, S_=560):
    f=SK[idx]; jo=f['joints']
    im=cv2.imread(cams[idx])
    pts={k:(int(v['x']),int(v['y'])) for k,v in jo.items()
         if v.get('conf',0)>0.25}
    if HUMAN_SKEL:
        for a,b_ in LINK:
            if a in pts and b_ in pts:
                cv2.line(im,pts[a],pts[b_],(120,125,135),5,cv2.LINE_AA)
        for k in KP:
            if k in pts:
                cv2.circle(im,pts[k],11,(255,255,255),-1,cv2.LINE_AA)
                cv2.circle(im,pts[k],11,(70,75,85),3,cv2.LINE_AA)
    # 크롭 중심은 스켈레톤 유무와 무관하게 키포인트 기준으로 유지
    _u=[pts[k] for k in KP if k in pts]
    xs=[p[0] for p in _u]; ys=[p[1] for p in _u]
    if xs:
        cx,cy=int(np.mean(xs)),int(np.mean(ys))
    else:
        cx,cy=im.shape[1]//2,im.shape[0]//2
    side=int(im.shape[0]*0.50)
    x0=int(np.clip(cx-side/2,0,im.shape[1]-side))
    y0=int(np.clip(cy-side/2,0,im.shape[0]-side))
    return cv2.resize(im[y0:y0+side,x0:x0+side],(S_,S_))

def robot(step, S_=560):
    p={k:HOME[k]+PN['profile'][k][step%NS] for k in J}
    hand,dv=apply(p); ge,ne,tip,b1,b2=hoe(hand,dv)
    rnd.update_scene(data,cam)
    im=cv2.cvtColor(rnd.render(),cv2.COLOR_RGB2BGR)
    bgc=im[3,3].astype(int)
    im[np.abs(im.astype(int)-bgc).sum(2)<BG_TOL]=255
    gl=[proj(np.array([x,0,GROUND])) for x in (-0.9,1.5)]
    if all(gl): cv2.line(im,gl[0],gl[1],(190,190,190),2)
    pg,phd,pne,pt,pb1,pb2=(proj(ge),proj(hand),proj(ne),proj(tip),proj(b1),proj(b2))
    if pg and phd: cv2.line(im,pg,phd,(110,110,115),12)
    if phd and pne: cv2.line(im,phd,pne,(95,100,110),7)
    if pt and pb1 and pb2:
        pol=np.array([pt,pb1,pb2],np.int32)
        cv2.fillPoly(im,[pol],(120,125,135)); cv2.polylines(im,[pol],True,(55,60,70),2)
    JB=['upper_arm_right','lower_arm_right','hand_right',
        'upper_arm_left','lower_arm_left','hand_left']
    q=[proj(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)].copy())
       for n in JB]
    for k in [(0,1),(1,2),(3,4),(4,5)]:
        if q[k[0]] and q[k[1]]: cv2.line(im,q[k[0]],q[k[1]],(120,125,135),3,cv2.LINE_AA)
    for c_ in q:
        if c_:
            cv2.circle(im,c_,9,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(im,c_,9,(70,75,85),2,cv2.LINE_AA)
    h,w=im.shape[:2]; side=h
    x0=int((w-side)/2)+int(w*0.02)
    return cv2.resize(im[:,max(0,x0):max(0,x0)+side],(S_,S_)), (GROUND-tip[2])*100

SZ=560; HDR=34; FTR=28
cols=[]; depths=[]
for k,(nm,step,ci) in enumerate(PHASE):
    hu=human(ci,SZ)
    ro,dep=robot(step,SZ)
    depths.append((nm,step,dep))
    hdr=np.full((HDR,SZ,3),255,np.uint8)
    cv2.putText(hdr,f'({chr(97+k)}) {nm}',(8,23),
                cv2.FONT_HERSHEY_SIMPLEX,0.62,(20,20,20),1,cv2.LINE_AA)
    ftr=np.full((FTR,SZ,3),255,np.uint8)
    cv2.putText(ftr,f'tool tip {"in soil" if dep>0 else "above soil"} '
                f'{abs(dep):.1f} cm',(8,19),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,(70,70,70),1,cv2.LINE_AA)
    cols.append(np.vstack([hdr,hu,ro,ftr]))
gap=np.full((cols[0].shape[0],10,3),255,np.uint8)
fig=cols[0]
for c in cols[1:]: fig=np.hstack([fig,gap,c])
cv2.imwrite(OUT,fig)
print(f'저장  {OUT}  {fig.shape[1]}x{fig.shape[0]}')
print('위상별 날끝 높이 (양수 = 흙 속)')
for nm,step,dep in depths:
    print(f'  {nm:6s} step {step:2d}   {dep:+6.1f} cm')
rnd.close()

VIEWER=shutil.which("eog") or shutil.which("xdg-open")
if VIEWER:
    subprocess.Popen([VIEWER,OUT],
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
