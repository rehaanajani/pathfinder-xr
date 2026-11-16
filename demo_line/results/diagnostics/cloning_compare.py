import sys, numpy as np, os
sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor
from pathfinder import data_tlio as D, metrics
from pathfinder.filter import FilterConfig, run
W = {}
def init():
    import torch; torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    W['p'] = Predictor('models/dispnet_faults/best.pt')
def job(sid):
    s = D.load(sid); out = {}
    for c in (False, True):
        e, _ = run(s, s.gyro[0], s.acc[0], None, None, W['p'], FilterConfig(cloning=c))
        out[c] = (metrics.ate(e, s.p, align=False), metrics.ate(e, s.p))
    return out
if __name__ == '__main__':
    with ProcessPoolExecutor(9, initializer=init) as ex:
        rows = list(ex.map(job, D.split('test')))
    for c in (False, True):
        raw = [r[c][0] for r in rows]; al = [r[c][1] for r in rows]
        print('cloning' if c else 'old    ', 'ATE raw mean %.2f median %.2f | yaw-aligned mean %.2f' % (np.mean(raw), np.median(raw), np.mean(al)))
