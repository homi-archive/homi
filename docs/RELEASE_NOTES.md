# Release notes

## Superseded figures

An earlier release of this dataset reported 95.1% five-fold
cross-validation accuracy for six-class activity recognition. That
figure came from a protocol that placed near-duplicate frames on both
sides of the fold boundary and included centroid coordinates among the
features. Neither is used here. The corresponding figures are 75.7%
under random folds and 55.7% under a temporal split, and the size of
that gap is itself reported in the paper.

## Re-annotation

The activity labels for the held-out session are a second pass. The
first was made before the gate was evaluated and adjusted afterwards,
which is not a sound basis for evaluation. The current pass was made
against the written rules in `datasets/HOMI/ANNOTATION.md`, from video
alone, with no cluster height, gate output or classifier probability
visible. Re-annotating changed 27% of frame labels but only 4.1% of the
binary ground-work target, and moved gate F1 by 0.2 points.
