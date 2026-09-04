#!/usr/bin/env python3
"""S8 고랑 위치 그라운드 트루스를 명시적으로 저장"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *
import json, collections

RUNS  = 'loc_runs_fine.json'
OUT   = 'loc_groundtruth_s8.json'
WRONG = {165:'F1', 592:'F2', 1078:'F1', 1139:'F1', 2128:'F2'}

runs = json.load(open(RUNS))
checks = {r['check'] for r in runs}
missing = set(WRONG) - checks
if missing:
    raise SystemExit(f'오류: WRONG 의 check {sorted(missing)} 가 {RUNS} 에 없음.\n'
                     f'  실제 check 값: {sorted(checks)}')

out, nwrong, nerr = [], 0, collections.Counter()
for r in runs:
    truth = WRONG.get(r['check'], r['pred'])
    n = r['end'] - r['start'] + 1
    ok = (truth == r['pred'])
    if not ok:
        nwrong += 1
        nerr[f"{r['pred']} -> {truth}"] += n
    out.append(dict(start=r['start'], end=r['end'], frames=n,
                    check=r['check'], pred=r['pred'], truth=truth,
                    correct=ok))

tot = sum(o['frames'] for o in out)
bad = sum(o['frames'] for o in out if not o['correct'])
json.dump({'session': 'S8',
           'annotator': 'author, from video',
           'n_segments': len(out),
           'n_segments_correct': len(out) - nwrong,
           'n_frames': tot,
           'n_frames_correct': tot - bad,
           'frame_accuracy': (tot - bad) / tot,
           'errors_by_pair': dict(nerr),
           'segments': out},
          open(OUT, 'w'), indent=1)

print(f'저장: {OUT}')
print(f'  구간 {len(out)-nwrong}/{len(out)} 정확')
print(f'  프레임 {tot-bad}/{tot} = {(tot-bad)/tot*100:.1f}%')
for k, v in sorted(nerr.items()):
    print(f'  {k}  {v}프레임')
