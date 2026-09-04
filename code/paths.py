"""데이터 경로. 각 스크립트 맨 위에서 `from paths import *` 로 부른다.

스크립트들은 파일명을 그대로 쓰므로, 여기서 데이터 폴더들을 현재 작업
디렉터리에 심볼릭 링크로 펼친다. 원본은 건드리지 않는다."""
import os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOMI = os.path.join(ROOT, 'datasets', 'HOMI')
WORK = os.path.join(ROOT, 'work')
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.join(WORK, 'results_figures'), exist_ok=True)

_ALIAS = {'s8_groundtruth.npy': 's8_groundtruth_v3.npy',
          's8_annotation.json': 's8_annotation_v4.json',
          's8_coco17.json':     's8_coco17_full.json',
          's8_3d.npz':          's8_3d_full.npz'}

for sub in ('derived', 'annotations', 'pose'):
    for src in glob.glob(os.path.join(HOMI, sub, '*')):
        name = _ALIAS.get(os.path.basename(src), os.path.basename(src))
        dst = os.path.join(WORK, name)
        if not os.path.exists(dst):
            os.symlink(src, dst)

# 원시 포인트클라우드는 S1..S8 이름으로 펼친다
_PCD = {'S1_empty_field':'S1','S2_standing':'S2','S3_walking':'S3',
        'S4_squatting':'S4','S5_squat_moving':'S5','S6_watering':'S6',
        'S7_homi_work':'S7','S8_mixed':'S8'}
_pd = os.path.join(WORK, 'pcd')
os.makedirs(_pd, exist_ok=True)
for long, short in _PCD.items():
    src = os.path.join(HOMI, 'pcd', long)
    dst = os.path.join(_pd, short)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)

os.chdir(WORK)
