#!/usr/bin/env python3
"""S8 파이프라인 3패널: 카메라 | LiDAR | 실시간 맵  /  아래 로봇"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, json, cv2, mujoco, os, glob, subprocess, shutil

# ══════ 레이아웃 ══════
W, H     = 800, 450          # 카메라, LiDAR 패널
MAPW     = 540               # 맵 패널 폭 (높이는 H)
RW, RH   = 1600, 700         # 로봇 렌더 크기 (Fig 6 와 공유하므로 고정)
PAD      = 100               # 정보 바
TRAIL_A  = 0.004             # 동선 누적 강도
DOT_R_M  = 0.20              # 현재 위치 점 반지름 (m)
EXCLUDED_ZONE = 'F4'
OUT      = 'results_figures/pipeline_S8_3panel.mp4'
# ═════════════════════

TOTW = W*2 + MAPW            # 2140

a,b = np.load('sync_model_S8.npy') if os.path.exists('sync_model_S8.npy') \
      else np.load('sync_model.npy')
gate = np.load('s8_gate_final_v2.npy')
gt   = np.load('s8_groundtruth_v3.npy', allow_pickle=True)
STD  = json.load(open('label_standard.json'))
LBL  = STD['paper_labels']

# ── 맵: 미리 저장한 배경과 궤적 ──
MAP_BASE = cv2.imread('map_base.png')
TRACK    = np.load('track_s8.npy')
ZONES    = np.load('zones_s8.npy')
MT       = json.load(open('map_transform.json'))
MC0, MC1 = MT['C0'], MT['C1']
MXR, MYR = MT['XR'], MT['YR']
MPPM, MMARGIN = MT['PPM'], MT['MARGIN']
MH_, MW_ = MAP_BASE.shape[:2]
print(f'맵 {MW_}x{MH_}, 궤적 {TRACK.shape}, CAM {MC0}..{MC1}')

def MT_(x, y):
    return (int(MMARGIN + (MYR[1]-y)*MPPM), int(MMARGIN + (MXR[1]-x)*MPPM))

ACC     = np.zeros((MH_, MW_), np.float32)
BLUE_LO = np.array([245,205,180], np.float32)
BLUE_HI = np.array([200, 40, 20], np.float32)
DOT_R   = max(6, int(DOT_R_M*MPPM))

# 프리미티브
S=json.load(open('homi_two_strokes.json'))
J=S['joints']; NS=S['profile_steps']
PN=S['strokes']['normal']; PF=S['strokes']['fine']
CYC_N=PN['cycle_s']; CYC_F=0.75
ABD_GAIN=0.273
FLEX_GAIN=1.0
HOME={'r_shoulder_y':118.0,'r_shoulder_x':0.0,'r_elbow_y':130.0,
      'l_shoulder_y':118.0,'l_shoulder_x':0.0,'l_elbow_y':130.0}

# 호미 형상
GRIP,NECK,BLADE,BHALF=0.0525,0.2385,0.149,0.0375
BEND,BANG=60.0,100.0
def hoe(hand,dv):
    up=np.array([0.0,0.0,1.0])
    perp=up-np.dot(up,dv)*dv
    n=np.linalg.norm(perp); perp=perp/n if n>1e-6 else up
    ang=np.radians(BEND)
    ax=dv*np.cos(ang)+perp*np.sin(ang); ax/=np.linalg.norm(ax)
    ge=hand-ax*GRIP; ne=hand+ax*NECK
    ab=np.radians(BEND-BANG)
    bx=dv*np.cos(ab)+perp*np.sin(ab); bx/=np.linalg.norm(bx)
    sd=np.cross(bx,up); ns=np.linalg.norm(sd)
    sd=sd/ns if ns>1e-6 else np.array([1.0,0,0])
    return ge,ne,ne+bx*BLADE,ne+sd*BHALF,ne-sd*BHALF

model=mujoco.MjModel.from_xml_path('humanoid.xml')
data=mujoco.MjData(model)
BID={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i):i
     for i in range(model.nbody)}
IDX={'root_z':2,'root_qw':3,'root_qy':5,'hip_y_r':12,'hip_y_l':18,
     'r_sh1':22,'r_sh2':23,'r_el':24,'l_sh1':25,'l_sh2':26,'l_el':27}
LEAN=np.radians(80); QW,QY=np.cos(LEAN/2),np.sin(LEAN/2)
MJ={'r_shoulder_y':('r_sh1',lambda x:-(x-90.0)),
    'r_shoulder_x':('r_sh2',lambda x:x),
    'r_elbow_y':('r_el',lambda x:-(180.0-x)),
    'l_shoulder_y':('l_sh1',lambda x:-(x-90.0)),
    'l_shoulder_x':('l_sh2',lambda x:-x),
    'l_elbow_y':('l_el',lambda x:-(180.0-x))}
LIM={'r_sh1':(-85,60),'r_sh2':(-85,60),'r_el':(-100,50),
     'l_sh1':(-85,60),'l_sh2':(-85,60),'l_el':(-100,50)}
def apply(p):
    data.qpos[:]=0
    data.qpos[IDX['root_z']]=0.64
    data.qpos[IDX['root_qw']]=QW; data.qpos[IDX['root_qy']]=QY
    data.qpos[IDX['hip_y_r']]=-LEAN; data.qpos[IDX['hip_y_l']]=-LEAN
    for k,(mj,fn) in MJ.items():
        lo,hi=LIM[mj]
        data.qpos[IDX[mj]]=np.radians(np.clip(fn(p[k]),lo,hi))
    mujoco.mj_forward(model,data)
    h=data.xpos[BID['hand_right']].copy()
    f=data.xpos[BID['lower_arm_right']].copy()
    dv=h-f; dv/=(np.linalg.norm(dv)+1e-9)
    return h,dv

_p0={}
for k in J:
    _v=PN['profile'][k][0]
    if 'shoulder_x' in k: _v*=ABD_GAIN
    elif 'shoulder_y' in k: _v*=FLEX_GAIN
    _p0[k]=HOME[k]+_v
h0,d0=apply(_p0)
GROUND=float(hoe(h0,d0)[2][2])
print(f'땅 높이 {GROUND*100:+.1f}cm')

# S8 위상/유형/강도
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
d8=np.load('s8_3d_full.npz'); Q8,FI8=d8['pose3d'],d8['frame_idx']
def ang3(p,q,r):
    v1,v2=p-q,r-q
    cs=np.sum(v1*v2,-1)/(np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1)+1e-9)
    return np.degrees(np.arccos(np.clip(cs,-1,1)))
sh8=savgol_filter(ang3(Q8[:,8],Q8[:,14],Q8[:,15]),7,2)
el8=savgol_filter(ang3(Q8[:,14],Q8[:,15],Q8[:,16]),7,2)
f0=1.0/((CYC_N+CYC_F)/2)
bb,aa=butter(3,[max(f0*0.55,0.3)/5.0,min(f0*2.2,4.2)/5.0],btype='band')
an=hilbert(filtfilt(bb,aa,el8))
um=np.roll(((np.angle(an)+np.pi)/(2*np.pi))%1.0,-8)
so=savgol_filter(np.abs(hilbert(filtfilt(bb,aa,sh8))),15,2)
eo=savgol_filter(np.abs(hilbert(filtfilt(bb,aa,el8))),15,2)
r_=so/(eo+1e-6)
lo_,hi_=np.percentile(r_,25),np.percentile(r_,80)
wN=np.clip((r_-lo_)/(hi_-lo_+1e-9),0,1)
amp=savgol_filter(np.abs(an),15,2)
al,ah=np.percentile(amp,20),np.percentile(amp,85)
inten=np.clip((amp-al)/(ah-al+1e-9),0.4,1.3)
phm={int(f):(float(um[i]),float(wN[i]),float(inten[i]))
     for i,f in enumerate(FI8)}

# ── 흰색 모노톤 ──
import mujoco as _mj
_HIDE=['floor','head','waist_lower','butt','thigh_right','shin_right',
       'foot1_right','foot2_right','thigh_left','shin_left',
       'foot1_left','foot2_left']
for _i in range(model.ngeom):
    _nm=_mj.mj_id2name(model,_mj.mjtObj.mjOBJ_GEOM,_i)
    model.geom_rgba[_i]=[1,1,1,0] if _nm in _HIDE else [0.86,0.86,0.88,1.0]
model.geom_matid[:]=-1
try: model.mat_texid[:]=-1
except Exception: pass
model.vis.headlight.ambient[:]=[0.7,0.7,0.7]
model.vis.headlight.diffuse[:]=[0.5,0.5,0.5]
model.vis.headlight.specular[:]=[0.0,0.0,0.0]
model.vis.rgba.fog[:]=[1,1,1,0]

rnd=mujoco.Renderer(model,height=RH,width=RW)
try:
    for _f in ('mjRND_SKYBOX','mjRND_FOG','mjRND_HAZE',
               'mjRND_SHADOW','mjRND_REFLECTION'):
        rnd.scene.flags[getattr(mujoco.mjtRndFlag,_f)]=0
except Exception as e:
    print('flag 설정 실패:',e)
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.distance,cam.azimuth,cam.elevation=1.55,102,-12
cam.lookat=np.array([0.20,0.0,GROUND+0.30])
def proj(q):
    az,e_=np.radians(cam.azimuth),np.radians(cam.elevation)
    fw=np.array([np.cos(e_)*np.cos(az),np.cos(e_)*np.sin(az),np.sin(e_)])
    eye=cam.lookat-fw*cam.distance
    rt=np.cross(fw,[0,0,1.0]); n=np.linalg.norm(rt)
    rt=rt/n if n>1e-6 else np.array([1.0,0,0]); up=np.cross(rt,fw)
    v=q-eye; z=np.dot(v,fw)
    if z<1e-3: return None
    f=RH/(2*np.tan(np.radians(45)/2))
    return (int(RW/2+f*np.dot(v,rt)/z),int(RH/2-f*np.dot(v,up)/z))

cam_all=sorted(glob.glob('video_S8_frames/*.jpg'))
C0=max(int(np.ceil((0-b)/a)),0)
C1=min(int(np.floor((2767-b)/a)),len(cam_all)-1)
print(f'LiDAR 유효 CAM {C0}~{C1}')

out=cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'),
                    10.0, (TOTW, H+RH+PAD))
trail=[]; ph=0.0
for ni in range(C0,C1+1):
    if (ni-C0)%300==0: print(f'  {ni-C0}/{C1-C0}')
    li=int(round(a*ni+b))

    # ── 게이트: 자세 AND 작업 구역 ──
    posture=bool(gate[ni]) if ni<len(gate) else False
    zi=ni-MC0
    zone=str(ZONES[zi]) if 0<=zi<len(ZONES) else '-'
    work_zone=(zone!='-') and (zone!=EXCLUDED_ZONE)
    on=posture and work_zone

    u,w_,g=phm.get(ni,(0.0,0.5,0.0))
    if not on: g=0.0
    cyc=CYC_F+(CYC_N-CYC_F)*w_
    ph=(ph+1.0/(cyc*10))%1.0
    i0=int(ph*NS)%NS
    p={}
    for k in J:
        dv_=(PF['profile'][k][i0]
             +(PN['profile'][k][i0]-PF['profile'][k][i0])*w_)*g
        if 'shoulder_x' in k: dv_*=ABD_GAIN
        elif 'shoulder_y' in k: dv_*=FLEX_GAIN
        p[k]=HOME[k]+dv_
    hand,dv=apply(p)
    ge,ne,tip,b1,b2=hoe(hand,dv)
    if on: trail.append(tip.copy())
    if len(trail)>16: trail.pop(0)

    # ── 1행 좌: 카메라, 중: LiDAR ──
    c=cv2.resize(cv2.imread(cam_all[ni]),(W,H))
    lp=f'vid_S8/f{li:05d}.png'
    l=cv2.resize(cv2.imread(lp),(W,H)) if os.path.exists(lp) \
      else np.full((H,W,3),255,np.uint8)

    # ── 1행 우: 맵 ──
    xy=TRACK[zi] if 0<=zi<len(TRACK) else np.array([np.nan,np.nan])
    if np.isfinite(xy[0]):
        lay=np.zeros((MH_,MW_),np.float32)
        cv2.circle(lay,MT_(*xy),DOT_R,1.0,-1,cv2.LINE_AA)
        ACC=np.clip(ACC+lay*TRAIL_A,0,1.0)
    mp=MAP_BASE.astype(np.float32)
    t=ACC[...,None]
    mp=mp*(1-t)+(BLUE_LO*(1-t)+BLUE_HI*t)*t
    mp=mp.astype(np.uint8)
    if np.isfinite(xy[0]):
        dc=(40,150,70) if on else (150,150,150)
        cv2.circle(mp,MT_(*xy),DOT_R+2,(215,160,190),-1)
        cv2.circle(mp,MT_(*xy),DOT_R+2,dc,3)
    mp=cv2.resize(mp,(MAPW,H))

    # ── 2행: 로봇 ──
    rnd.update_scene(data,cam)
    sim=cv2.cvtColor(rnd.render(),cv2.COLOR_RGB2BGR)
    bgc=sim[3,3].astype(int)
    sim[np.abs(sim.astype(int)-bgc).sum(2)<90]=255
    gl=[proj(np.array([x,0,GROUND])) for x in (-0.9,1.5)]
    if all(gl): cv2.line(sim,gl[0],gl[1],(190,190,190),2)
    for q in trail:
        pq=proj(q)
        if pq: cv2.circle(sim,pq,3,(70,150,215),-1)
    pg,phd,pne,pt,pb1,pb2=(proj(ge),proj(hand),proj(ne),
                           proj(tip),proj(b1),proj(b2))
    if pg and phd: cv2.line(sim,pg,phd,(110,110,115),12)
    if phd and pne: cv2.line(sim,phd,pne,(95,100,110),7)
    if pt and pb1 and pb2:
        pol=np.array([pt,pb1,pb2],np.int32)
        cv2.fillPoly(sim,[pol],(120,125,135))
        cv2.polylines(sim,[pol],True,(55,60,70),2)
    _O=np.array([0.0,0.0,GROUND]); _L=0.12
    for _v,_c,_t in [([_L,0,0],(60,60,220),'x'),
                     ([0,_L,0],(60,170,60),'y'),
                     ([0,0,_L],(220,120,60),'z')]:
        _q0=proj(_O); _q1=proj(_O+np.array(_v))
        if _q0 and _q1:
            cv2.arrowedLine(sim,_q0,_q1,_c,1,cv2.LINE_AA,tipLength=0.20)
            cv2.putText(sim,_t,(_q1[0]+6,_q1[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,_c,1,cv2.LINE_AA)
    JB=['upper_arm_right','lower_arm_right','hand_right',
        'upper_arm_left','lower_arm_left','hand_left']
    pts=[proj(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,_n)].copy())
         for _n in JB]
    for k in [(0,1),(1,2),(3,4),(4,5)]:
        if pts[k[0]] and pts[k[1]]:
            cv2.line(sim,pts[k[0]],pts[k[1]],(120,125,135),3,cv2.LINE_AA)
    for q in pts:
        if q:
            cv2.circle(sim,q,9,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(sim,q,9,(70,75,85),2,cv2.LINE_AA)
    # 로봇 렌더는 1600 폭, 캔버스는 2140 폭이므로 가운데 배치
    robot=np.full((RH,TOTW,3),255,np.uint8)
    _x0=(TOTW-RW)//2
    robot[:,_x0:_x0+RW]=sim

    col=(40,150,70) if on else (150,150,150)
    if on:
        for panel in (c,l,mp):
            cv2.rectangle(panel,(0,0),(panel.shape[1]-1,panel.shape[0]-1),col,4)

    # ── 정보 바 ──
    bar=np.full((PAD,TOTW,3),255,np.uint8)
    cv2.putText(bar,f'CAM {ni}   t={ni/10:.1f}s   {LBL.get(gt[ni],gt[ni])}',
                (14,28),cv2.FONT_HERSHEY_SIMPLEX,0.60,(30,30,30),1,cv2.LINE_AA)
    cv2.putText(bar,f'LiDAR {li}   gate {"ON" if on else "off"}',
                (W+14,28),cv2.FONT_HERSHEY_SIMPLEX,0.60,col,2,cv2.LINE_AA)
    lab=('normal stroke' if w_>0.6 else 'fine stroke' if w_<0.4 else 'blend')
    dep=(GROUND-tip[2])*100
    txt=f'{lab}   ' + (f'soil {dep:.1f} cm' if dep>0 else f'air {-dep:.1f} cm')
    cv2.putText(bar,txt if on else 'idle',(2*W+14,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.60,col,2,cv2.LINE_AA)

    zcol=(40,150,70) if work_zone else (60,110,180)
    ztxt=f'furrow {zone}'+('' if zone=='-' else
         (' (work zone)' if work_zone else ' (excluded)'))
    cv2.putText(bar,ztxt,(14,58),cv2.FONT_HERSHEY_SIMPLEX,0.56,
                zcol,1,cv2.LINE_AA)
    cv2.putText(bar,f'posture {"ON" if posture else "off"}'
                    f'   AND   zone {"ON" if work_zone else "off"}'
                    f'   ->   camera {"ON" if on else "off"}',
                (W+14,58),cv2.FONT_HERSHEY_SIMPLEX,0.56,col,1,cv2.LINE_AA)
    if np.isfinite(xy[0]):
        cv2.putText(bar,f'x {xy[0]:.2f}  y {xy[1]:+.2f}',(2*W+14,58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.56,(60,60,60),1,cv2.LINE_AA)

    for _t,_x in [('1. camera',14),('2. LiDAR gate',W+14),
                  ('3. plot position',2*W+14)]:
        cv2.putText(bar,_t,(_x,PAD-14),cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,(140,140,140),1,cv2.LINE_AA)
    cv2.putText(bar,'4. robot playback',(TOTW//2-70,PAD-14),
                cv2.FONT_HERSHEY_SIMPLEX,0.44,(140,140,140),1,cv2.LINE_AA)

    out.write(np.vstack([np.hstack([c,l,mp]), robot, bar]))
out.release(); rnd.close()
print(f'완료: {OUT}  {TOTW}x{H+RH+PAD}')

V=shutil.which('vlc') or shutil.which('mpv') or shutil.which('xdg-open')
if V: subprocess.Popen([V,OUT],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
