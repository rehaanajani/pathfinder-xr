"""Ray-cast renderer of a textured box room through a calibrated camera model.

Test and smoke-fixture only: it lets the visual front end be checked against exactly known motion
with the real projection model (equidistant fisheye or rational radtan) and needs no dataset.
"""

import numpy as np

from .camera_model import CameraModel


_TEX = {}
TEXEL_M = 0.01           # 1 cm texels, 8 m x 8 m per face (non-repeating, so no stereo aliasing)


def _tile(seed):
    if seed not in _TEX:
        import cv2
        rng = np.random.default_rng(seed)
        n = 800
        img = np.zeros((n, n), np.float32)
        for sigma, w in ((2.0, 1.0), (6.0, 1.5), (18.0, 2.0)):
            img += w * cv2.GaussianBlur(rng.normal(size=(n, n)).astype(np.float32), (0, 0), sigma) * sigma
        img = (img - img.mean()) / img.std()
        _TEX[seed] = np.clip(128 + 55 * img, 0, 255).astype(np.float32)
    return _TEX[seed]


def _texture(u, v, seed):
    """Bilinear sample of a fixed random multi-scale texture; u, v in metres on the face."""
    tex = _tile(seed)
    n = tex.shape[0]
    x = np.clip(u / TEXEL_M + n / 2, 0, n - 1.001)
    y = np.clip(v / TEXEL_M + n / 2, 0, n - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * tex[y0, x0] + fx * (1 - fy) * tex[y0, x0 + 1]
            + (1 - fx) * fy * tex[y0 + 1, x0] + fx * fy * tex[y0 + 1, x0 + 1])


_RAYS = {}


def _rays(cam: CameraModel):
    if cam not in _RAYS:
        vv, uu = np.mgrid[0:cam.height, 0:cam.width].astype(float)
        _RAYS[cam] = cam.unproject(np.stack([uu.ravel(), vv.ravel()], 1))
    return _RAYS[cam]


def render(cam: CameraModel, T_w_c: np.ndarray, half=(3.0, 3.0, 1.5), seed: int = 0) -> np.ndarray:
    """uint8 image of the room interior [-hx, hx] x [-hy, hy] x [-hz, hz] seen from pose T_w_c."""
    xy = _rays(cam)
    ok = np.isfinite(xy).all(1)
    d_c = np.c_[np.nan_to_num(xy), np.ones(len(xy))]
    d = d_c @ T_w_c[:3, :3].T
    o = T_w_c[:3, 3]
    h = np.asarray(half, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(d > 0, (h - o) / d, (-h - o) / d)
    axis = np.argmin(np.where(tt > 0, tt, np.inf), 1)
    tmin = tt[np.arange(len(tt)), axis]
    P = o + d * tmin[:, None]
    img = np.zeros(len(P))
    for a in range(3):
        for side in (False, True):
            m = (axis == a) & ((P[:, a] > 0) == side)
            u_, v_ = [P[m, j] for j in range(3) if j != a]
            img[m] = _texture(u_, v_, seed + 10 * a + int(side))
    img[~ok] = 0
    return img.reshape(cam.height, cam.width).astype(np.uint8)
