# Code

All paths are relative to the repository root. Run from `code/` with the
dataset directories alongside, or set the paths at the top of each script.

## Dependencies

    numpy scipy scikit-learn torch open3d opencv-python matplotlib mujoco

## What produces what

| Script | Produces | Paper |
|---|---|---|
| `cache_pcd_native.py` | `pcd_cache_native.npz` | PointNet input at native return density |
| `gate_pointnet_native.py` | `s8_score_pn_native.npy` | PointNet row of the gate comparison |
| `gate_final.py` | `s8_score_raw.npy`, `s8_score_final.npy`, `s8_gate_final_v2.npy`, `gate_final_results.json` | Gate figures in Sec. V-B |
| `duty_sweep.py` | `duty_sweep.pdf`, `duty_sweep_results.json` | Fig. 3, duty-cycle sweep |
| `split_half.py` | `split_half_results.json` | Split-half threshold check, Sec. V-B |
| `event_metrics.py` | `event_metrics.json` | Event-level metrics, Sec. V-B |
| `ablation.py` | `ablation_results.json` | Table II, feature and smoothing ablation |
| `make_map_w6.py` | plot geometry and per-frame position (imported by others) | Sec. V-C |
| `gate_location.py` | `furrow_s8.npy`, `gate_s8_*.npy`, `gate_location_results.json` | Position gate, Sec. V-C |
| `save_loc_groundtruth.py` | `loc_groundtruth_s8.json` | 29-interval position ground truth |
| `save_track_s8.py` | `track_s8.npy`, `map_base.png`, `map_transform.json` | Cumulative track panel of Fig. 4 |
| `annotate_s8.py` | `s8_annotation.json`, `s8_groundtruth.npy` | Keyboard annotation tool, rules in `ANNOTATION.md` |
| `make_fig2b.py`, `make_fig2_compose.py` | `fig2_panels.png` | Fig. 1(b), plot layout |
| `make_fig5_final.py` | `fig5_timeline.pdf` | Fig. 4, gate timeline and track |
| `make_fig6_final3.py` | `fig6_strokes.png` | Fig. 5, stroke phases |
| `render_pipeline_final2.py` | robot rendering settings (imported by `make_fig6_final3.py`) | |
| `render_pipeline_3panel.py` | `pipeline_S8_3panel.mp4` | Supplementary video |

## Order

Everything below `gate_final.py` depends on it, so run that first if you
are starting from the released derived arrays.

    python3 gate_final.py          # gate scores and decisions
    python3 gate_location.py       # furrow assignment (imports make_map_w6.py)
    python3 save_loc_groundtruth.py
    python3 save_track_s8.py
    python3 duty_sweep.py
    python3 split_half.py
    python3 event_metrics.py
    python3 ablation.py

To rebuild from raw point clouds instead, run `cache_pcd_native.py` and
`gate_pointnet_native.py` first; both read `datasets/HOMI/pcd/`.

## Notes

`make_map_w6.py` and `render_pipeline_final2.py` are imported by other
scripts through `exec`, which is how the plot geometry and the robot
camera settings are shared. They can also be run directly.

`annotate_s8.py` shows only video frames. It deliberately does not
display cluster height, gate output or classifier probability, so that
annotation stays independent of model behaviour.
