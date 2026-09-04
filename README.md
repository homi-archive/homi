# HOMI

A LiDAR-gated sensing pipeline for capturing manual agricultural work,
with the dataset it was built on.

A single fixed LiDAR at the edge of a plot decides when a worker is
engaged in ground work and enables a camera only during those intervals.
The camera supplies joint-level motion; the LiDAR supplies both the
decision and, from the same returns, where in the plot the work happened.
Between them the session yields a record of what was done and where,
with imagery existing for less than half of it.

This repository holds the dataset, the annotations, and the code that
produces every number and figure in the accompanying paper.

## Layout

    datasets/HOMI/     the dataset (see its own README)
    code/              analysis and figure scripts (see code/README.md)
    models/            trained gate model
    docs/              release notes

## Quick start

    cd code
    python3 gate_final.py

reads the released feature arrays and reproduces the gate scores and
decisions. `code/README.md` lists which script produces which table and
figure, and the order to run them in.

## What the pipeline does

**Gate.** A Random Forest over thirteen per-frame features, ten
describing cluster shape and three describing how the cluster centroid
moves over a two-second window, is smoothed by a causal trailing mean
and thresholded. The target is ground work in general rather than any
specific task, because at twenty to fifty returns on the worker the
sensor reads posture, not what is in the hand.

**Position.** The same returns carry the ridges and furrows. Registering
the plot once against the background model lets each frame be placed in
a furrow, so where and for how long the work happened is recorded
without the camera being enabled.

**Pose and retargeting.** Within gated intervals, YOLOv8-pose and
MotionBERT give six joint-angle trajectories, which are retargeted in
simulation to a fixed-posture upper-body arm. The robot has not been
built; everything in that part is measured in simulation.

## Reproducibility notes

The annotation of the held-out session is a second pass, made against
written rules and without reference to model output. Both the rules and
the interval boundaries are released so the labelling can be audited.

Thresholds are checked two ways: swept across camera duty cycles from 40
to 70%, and chosen on the first half of the held-out session and
evaluated on the second.

Centroid coordinates are excluded from the features on purpose. Adding
them raises cross-validation accuracy and destroys held-out performance,
because each scripted scenario was recorded in a different part of the
plot.

## Licensing

Code: MIT. Datasets: CC BY 4.0.

## Citation

Reference to be added on publication.
