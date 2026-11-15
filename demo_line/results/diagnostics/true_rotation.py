import sys, numpy as np
sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation as R
from pathfinder import data_tlio as D, metrics
from pathfinder.filter import FilterConfig, run
W = {}
def init():
    import torch; torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    W['p'] = Predictor('models/dispnet_faults/best.pt')
def job(sid):
    s = D.load(sid); P = W['p']
    Rgt = R.from_quat(s.q).as_matrix()
    class GTRot:
        window, every = 200, 20
        def __init__(self): self.k = 200
        def __call__(self, win):
            k = self.k; self.k += 20
            gw = np.einsum('nij,nj->ni', Rgt[k-200:k], s.gyro[0][k-200:k]); aw = np.einsum('nij,nj->ni', Rgt[k-200:k], s.acc[0][k-200:k])
            return P(np.concatenate([gw, aw], 1))
    # pure network prediction quality along the run, with GT rotation, summed open-loop
    e1, _ = run(s, s.gyro[0], s.acc[0], None, None, P, FilterConfig())
    e2, _ = run(s, s.gyro[0], s.acc[0], None, None, GTRot(), FilterConfig())
    # tilt error of the filter: compare gravity direction
    return {'filter_rotation': metrics.ate(e1, s.p, align=False), 'true_rotation': metrics.ate(e2, s.p, align=False)}
if __name__ == '__main__':
    with ProcessPoolExecutor(9, initializer=init) as ex:
        rows = list(ex.map(job, D.split('test')[:18]))
    for k in rows[0]:
        print(f'{k:16s} ATE mean {np.mean([r[k] for r in rows]):.2f} m  median {np.median([r[k] for r in rows]):.2f} m')
