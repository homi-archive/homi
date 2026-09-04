#!/usr/bin/env python3
"""Fig 2(b): 두둑/고랑 배치도. 등방 배율, 캔버스는 배율에서 역산"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import numpy as np, cv2
src=open('make_map_w6.py').read()
exec(src[:src.index("XR=(0.30,4.80)")], globals())

# ── 크기 ──
PPM        = 155.0     # px per metre. 이 값만 올리면 그래픽이 커짐 (이전 상태 = 125)
FONT_EXTRA = 1.0       # 글자만 추가로 키울 때. 인쇄 크기 안 맞으면 2.0 부터
ML,MR,MT,MB = 30,30,20,20   # 캔버스 여백 px

SHOW_ARCS  = True
LABEL_HALO = True

FONT = cv2.FONT_HERSHEY_SIMPLEX
XR=(-0.02,4.32)                       # x: 화면 세로
YR=(YLO-0.10, YHI+0.10)               # y: 화면 가로
SPAN_X, SPAN_Y = XR[1]-XR[0], YR[1]-YR[0]

S = PPM
DRAW_W, DRAW_H = int(SPAN_Y*S), int(SPAN_X*S)
W, H = DRAW_W+ML+MR, DRAW_H+MT+MB
FONT_K = (S/125.0)*FONT_EXTRA         # 글자를 배율에 묶음
K = S/125.0                           # 기호 크기도 같이

def T(x,y):
    return (int(ML+(YR[1]-y)*S), int(MT+(XR[1]-x)*S))

def text(txt,px,py,sc=0.46,c=(50,50,50),th=1,anchor='l',halo=0):
    sc*=FONT_K; th=max(1,int(round(th*FONT_K)))
    (tw,tht),_=cv2.getTextSize(txt,FONT,sc,th)
    x = px if anchor=='l' else (px-tw if anchor=='r' else px-tw//2)
    org=(int(x),int(py+tht//2))
    if halo:
        cv2.putText(img,txt,org,FONT,sc,(255,255,255),th+halo,cv2.LINE_AA)
    cv2.putText(img,txt,org,FONT,sc,c,th,cv2.LINE_AA)

img=np.full((H,W,3),255,np.uint8)

# ── 고랑, 두둑 ──
for f in FU+[F6]:
    cv2.rectangle(img,T(f['xhi'],f['yhi']),T(f['xlo'],f['ylo']),(249,249,249),-1)
for r in RG:
    cv2.rectangle(img,T(r['x1'],r['hi']),T(r['x0'],r['lo']),(220,220,220),-1)
    cv2.rectangle(img,T(r['x1'],r['hi']),T(r['x0'],r['lo']),(180,180,180),1)

# ── 두둑, 고랑 라벨 ──
for r in RG:
    u,v=T((r['x0']+r['x1'])/2,(r['lo']+r['hi'])/2)
    text(r['name'],u,v,0.6,(95,95,95),2,'c')
for f in FU:
    u,v=T((f['xlo']+f['xhi'])/2,(f['ylo']+f['yhi'])/2)
    text(f['name'],u,v,0.5,(40,40,40),2,'c',halo=3 if LABEL_HALO else 0)
u,v=T((F6['xlo']+F6['xhi'])/2,(F6['ylo']+F6['yhi'])/2)
text('F6',u,v,0.5,(40,40,40),2,'c',halo=3 if LABEL_HALO else 0)

# ── 지면 반사 한계 호: 캔버스 안으로 클리핑 ──
if SHOW_ARCS:
    for R_ in (1.79,3.84):
        pts=[T(R_*np.cos(t),R_*np.sin(t)) for t in np.linspace(-1.05,1.5,400)]
        pts=[p for p in pts if ML<=p[0]<W-MR and MT<=p[1]<H-MB]
        for k in range(0,len(pts)-1,3):
            cv2.line(img,pts[k],pts[k+1],(200,200,200),1,cv2.LINE_AA)

# ── 센서 ──
LY=RG[1]['hi']-0.08; CY=LY-0.30
lu,lv=T(0.06,LY); cu,cv_=T(0.06,CY)
cv2.circle(img,(lu,lv),max(4,int(7*K)),(35,35,35),-1)
sq=max(4,int(6*K))
cv2.rectangle(img,(cu-sq,cv_-sq),(cu+sq,cv_+sq),(110,110,110),-1)
text('LiDAR',lu,lv-int(22*K),0.46,(35,35,35),1,'c')
text('camera',cu,cv_+int(26*K),0.46,(110,110,110),1,'c')

# ── 물통 (지름 1 m) ──
tc=T(0.06,1.90); tr=max(8,int(0.50*S))
cv2.circle(img,tc,tr,(238,238,238),-1)
cv2.circle(img,tc,tr,(178,178,178),1)
text('tank',tc[0]-tr-int(10*K),tc[1],0.44,(150,150,150),1,'r')

# ── 축척 막대: 캔버스 안쪽 오른쪽 아래 ──
BAR_M=1.0; bar=int(BAR_M*S)
inset=int(14*K)
sb=(W-MR-inset, H-MB-inset); sa=(sb[0]-bar, sb[1])
cv2.line(img,sa,sb,(50,50,50),2)
for p in (sa,sb): cv2.line(img,(p[0],p[1]-int(5*K)),(p[0],p[1]+int(5*K)),(50,50,50),2)
text(f'{BAR_M:.0f} m',(sa[0]+sb[0])//2,sa[1]-int(14*K),0.46,(50,50,50),1,'c')

cv2.imwrite('results_figures/fig2b_map.png',img)
print(f'저장  {W}x{H}   배율 {S:.1f} px/m  (등방)')
print(f'  그림 영역 {DRAW_W} x {DRAW_H} px')
print(f'  두둑 {len(RG)}개, 고랑 {len(FU)}개 + F6')
