# Release notes

## Superseded material

An earlier version of this repository reported a 95.1% five-fold
cross-validation accuracy for six-class activity recognition. That
figure came from an evaluation protocol that placed near-duplicate
frames on both sides of the fold boundary and included centroid
coordinates among the features. Neither is used here. The
corresponding figures in the paper are 75.7% under random folds and
55.7% under a temporal split, and the reasons for the gap are quantified
in the paper.

The activity labels for the held-out session were also re-annotated.
The earlier pass was made before the gate was evaluated and adjusted
afterwards; the current pass was made against written rules with no
model output visible.
