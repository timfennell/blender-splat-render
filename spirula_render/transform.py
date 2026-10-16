"""Moving splats from object space to world space (several splat objects share one render)."""

import numpy as np

from .relight import sh_basis


def decompose(M):
    """4x4 object->world -> (rotation 3x3, uniform scale, translation, non-uniform?)."""
    M = np.asarray(M, np.float64)
    A = M[:3, :3]
    s_axes = np.linalg.norm(A, axis=0)
    s = float(np.cbrt(abs(np.linalg.det(A)))) or 1.0
    R = A / np.maximum(s_axes, 1e-12)
    U, _, Vt = np.linalg.svd(R)                 # nearest proper rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    nonuniform = float(np.max(s_axes) / max(np.min(s_axes), 1e-12)) > 1.001
    return R, s, M[:3, 3], nonuniform


def quat_from_matrix(R):
    """w, x, y, z."""
    m = R
    t = np.trace(m)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q, np.float64)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """a (4,) times each row of b (N, 4); both w, x, y, z."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1).astype(np.float32)


_BANDS = ((0, 3), (3, 8), (8, 15))


def rotate_sh(sh, R):
    """sh [N,K,3] (coefficient-major, bands 1..3) for splats rotated by R: colour seen along d
    becomes the old colour along R^T d. Per-band matrices by least squares over directions."""
    if sh is None:
        return None
    K = sh.shape[1]
    rng = np.random.default_rng(12345)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Bd = sh_basis(d, 15)[:, 1:]
    Brd = sh_basis(d @ R, 15)[:, 1:]                 # rows are R^T d
    out = sh.copy()
    for lo, hi in _BANDS:
        if hi > K:
            break
        X = np.linalg.lstsq(Bd[:, lo:hi], Brd[:, lo:hi], rcond=None)[0]   # basis(R^T d) = basis(d) @ X
        out[:, lo:hi, :] = np.einsum('km,nmc->nkc', X.astype(np.float32), sh[:, lo:hi, :])
    return out


def to_world(d, M):
    """Object-space splat arrays (scene_data.splat_arrays) -> world space."""
    R, s, t, nonuniform = decompose(M)
    out = dict(d)
    out["means"] = (d["means"] @ (R * s).T.astype(np.float32) + t.astype(np.float32)).astype(np.float32)
    if not np.allclose(R, np.eye(3), atol=1e-7):
        out["quats"] = quat_mul(quat_from_matrix(R), d["quats"])
        out["sh"] = rotate_sh(d["sh"], R)
    out["log_scales"] = (d["log_scales"] + np.float32(np.log(s))).astype(np.float32)
    out["nonuniform"] = nonuniform
    return out


def merge(parts):
    """Concatenate world-space splat sets; SH padded to the highest degree among them."""
    K = max((0 if p["sh"] is None else p["sh"].shape[1]) for p in parts)
    degree = int(round(np.sqrt(K + 1))) - 1
    out = {}
    for name in ("means", "quats", "log_scales", "logit_opacities", "dc"):
        out[name] = np.ascontiguousarray(np.concatenate([p[name] for p in parts]), np.float32)
    if K:
        shs = []
        for p in parts:
            sh = p["sh"] if p["sh"] is not None else np.zeros((p["n"], 0, 3), np.float32)
            if sh.shape[1] < K:
                sh = np.concatenate([sh, np.zeros((p["n"], K - sh.shape[1], 3), np.float32)], 1)
            shs.append(sh)
        out["sh"] = np.ascontiguousarray(np.concatenate(shs), np.float32)
    else:
        out["sh"] = None
    out["degree"] = degree
    out["n"] = len(out["means"])
    return out
