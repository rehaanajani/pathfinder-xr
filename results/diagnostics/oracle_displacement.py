import sys, numpy as np
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
    s = D.load(sid)
    class Oracle:
        window, every = 200, 20
        def __init__(self, std): self.k, self.std = 200, std
        def __call__(self, win):
            k = self.k; self.k += 20
            d = s.p[k] - s.p[k - 200]
            return d + np.random.default_rng(k).normal(0, self.std, 3), np.eye(3) * max(self.std, 0.02) ** 2
    out = {}
    e, _ = run(s, s.gyro[0], s.acc[0], None, None, W['p'], FilterConfig()); out['network'] = metrics.ate(e, s.p, align=False)
    for std in (0.0, 0.05, 0.136):
        e, _ = run(s, s.gyro[0], s.acc[0], None, None, Oracle(std), FilterConfig()); out[f'true+noise{std}'] = metrics.ate(e, s.p, align=False)
    return out
if __name__ == '__main__':
    with ProcessPoolExecutor(9, initializer=init) as ex:
        rows = list(ex.map(job, D.split('test')[:18]))
    for k in rows[0]:
        print(f'{k:18s} ATE mean {np.mean([r[k] for r in rows]):.2f} m  median {np.median([r[k] for r in rows]):.2f} m')
