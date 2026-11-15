"""Shared fixtures. The displacement oracle reads the reference trajectory: DIAGNOSTIC ONLY, it isolates
filter behaviour from network quality and is never part of a scored or production estimator input."""

import numpy as np

from pathfinder import pipeline
from pathfinder.filter import run


def displacement_oracle(seq, std=0.05, window=200, every=20):
    """Perfect learned displacement (reference-based, labelled diagnostic)."""
    def predictor(win):
        k = predictor.k
        predictor.k += every
        return seq.p[k] - seq.p[k - window], np.eye(3) * std ** 2
    predictor.k, predictor.window, predictor.every = window, window, every
    predictor.label = "oracle_diagnostic"
    return predictor


def run_shipped(seq, faults, seed=0, predictor=None, init="truth", cam_transform=None, cfg=None):
    """The shipped path: ImuMonitor + pipeline.ours_cfg() through the streaming estimator."""
    bad, cam = pipeline.prepare(seq, faults, seed)
    if cam_transform is not None:
        cam = cam_transform(cam)
    fg, fa, info = pipeline.fuse_imus(bad)
    est, log, out = run(seq, fg, fa, None, cam, predictor, cfg or pipeline.ours_cfg(),
                        imu_quality=info["quality"], init=init, return_outputs=True)
    return est, log, out, info


def err(est, seq):
    return np.linalg.norm(est - seq.p, axis=1)
