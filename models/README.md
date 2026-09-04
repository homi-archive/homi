# Trained models

| File | Contents |
|---|---|
| `gate_rf.joblib` | Random Forest gate. A dict holding the fitted `sklearn` model, the feature-column index, the relative-motion window and radius, the smoothing window and the 0.50 threshold. |

To use it:

    import joblib, numpy as np
    m = joblib.load('models/gate_rf.joblib')
    p = m['model'].predict_proba(X)[:, m['classes'].index('target')]

`X` must be the ten shape features selected by `m['keep_idx']`
concatenated with the three relative-motion features, in that order.
`code/gate_final.py` shows how both are computed.

The score then needs the causal trailing mean over `m['w_smooth']`
frames before thresholding at `m['threshold']`; the gate is not the raw
per-frame probability.

The LSTM and PointNet baselines are not shipped as weights. Their
per-frame scores on the held-out session are in
`datasets/HOMI/derived/`, and `code/gate_pointnet_native.py` retrains
PointNet from the released caches in a few minutes.
