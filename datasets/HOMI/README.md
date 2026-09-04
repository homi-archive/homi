# HOMI

Ground-fixed LiDAR recording of hand-tool agricultural work, with a
synchronised camera used only to derive pose. 16,676 point-cloud frames
across eight scenarios, recorded in a single session on 22 April 2026 at
a citizens' allotment in Suwon, Republic of Korea.

One participant performed all activities. She is long experienced with
the Ho-Mi, a short-handled Korean hand tool, and took part with informed
consent after reviewing the recorded material.

## Sensors

| | |
|---|---|
| LiDAR | Unitree L2, tripod-mounted, base 112 cm above ground, tilted 16.8° down |
| | 16 scan lines per revolution, ~2 cm range accuracy |
| | Effective rate 12 Hz, established by frame correspondence rather than from the specification |
| Camera | iPhone 13 Pro, decimated to 10 fps, elevated oblique view |

The two devices were started and stopped independently, so each scenario
carries its own alignment. `derived/sync_model*.npy` holds the fitted
`(slope, intercept)` mapping camera frame index to LiDAR frame index.

Ground returns from the plot span 179 to 384 cm from the sensor; the
downward tilt leaves soil nearer than 179 cm unobserved, though a
standing person at that distance is still intercepted.

## Scenarios

| Dir | Activity | Frames |
|---|---|---|
| `S1_empty_field` | background reference | 2,185 |
| `S2_standing` | standing | 2,167 |
| `S3_walking` | walking | 2,188 |
| `S4_squatting` | crouched hand weeding, fixed position | 2,154 |
| `S5_squat_moving` | the same while shifting along the ridge | 1,765 |
| `S6_watering` | spraying with a hand-held sprayer, standing | 1,588 |
| `S7_homi_work` | Ho-Mi work | 1,861 |
| `S8_mixed` | unscripted mixed session, held out for evaluation | 2,768 |

S1 through S7 are scripted single-activity sequences and carry one label
each. S8 is annotated frame by frame; see `ANNOTATION.md` for the rules
and `annotations/` for the labels.

## Layout

    pcd/                    raw point clouds, one .pcd per frame
    derived/                features, caches and calibration
    annotations/            S8 frame labels, furrow assignment, position ground truth
    pose/                   2D keypoints and lifted 3D for S8
    ANNOTATION.md           annotation rules used for S8

### derived/

| File | Contents |
|---|---|
| `features_X_v4.npy` | per-frame cluster features, 8,830 × 13 |
| `features_y_std.npy` | scenario labels for the above |
| `features_frame_v4.npy`, `features_scene_v4.npy` | frame index and scenario id |
| `feature_keep_idx.npy` | which columns are the ten shape features |
| `s8_feat_v4.npy`, `s8_frames_v4.npy` | the same for S8 |
| `pcd_cache_native.npz` | 24-point clusters with padding mask, for PointNet |
| `s8_pcd_cache_native.npz` | the same for S8 |
| `background_model.npy` | reference model accumulated over S1 |
| `plot_layout.json` | surveyed ridge boundaries |
| `sync_model*.npy` | camera-to-LiDAR frame mapping |
| `s8_score_lstm_bin.npy`, `s8_score_pn_native.npy` | per-frame scores from the two comparison models |

Centroid coordinates are deliberately absent from the feature set.
Including them raises six-class cross-validation accuracy but collapses
held-out gate performance, because each scripted scenario was recorded
in a different part of the plot.

### annotations/

| File | Contents |
|---|---|
| `s8_groundtruth.npy` | per-frame activity label, 2,498 entries |
| `s8_annotation.json` | the interval boundaries the labels were built from |
| `label_standard.json` | label names and which count as ground work |
| `furrow_s8.npy` | occupied furrow per frame, camera 106–2412 |
| `gate_s8_camera_idx.npy` | camera frame indices for the above |
| `loc_groundtruth_s8.json` | 29 intervals checked against video for furrow accuracy |
| `loc_runs_fine.json` | the raw interval list those checks were made on |

The activity labels are a second pass. An earlier pass was made before
the gate was evaluated; re-annotating against written rules changed 27%
of frame labels but only 4.1% of the binary ground-work target. Both the
interval boundaries and the rules are included so the labelling can be
audited.

### pose/

| File | Contents |
|---|---|
| `s8_coco17.json` | YOLOv8n-pose output on S8, 17 COCO keypoints with confidence, per camera frame |
| `s8_3d.npz` | MotionBERT lifting of the above, 17 joints × 3, root-relative |
| `yolo_skeleton_s7.json` | YOLOv8n-pose on S7, the detection-rate comparison |
| `mediapipe_purple_crop.json` | MediaPipe on the same S7 crops, for the same comparison |
| `homi_two_strokes.json` | two 24-step joint-angle profiles used for retargeting |

`s8_coco17.json` marks frames where detection failed with
`valid: false`. `s8_3d.npz` is continuous because gaps are filled by
linear interpolation before lifting; roughly 12% of the Ho-Mi frames
used in the paper are interpolated rather than detected.

## Not included

Video frames are not released. The camera recording contains the
participant and other allotment users, and the pose stage is fully
represented by `pose/s8_coco17.json`, which carries no imagery.

## Licence

Data: CC BY 4.0. Code in `code/`: MIT.
