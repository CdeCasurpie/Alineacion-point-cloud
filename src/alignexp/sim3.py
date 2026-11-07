import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def apply(S, points):
    return np.asarray(points) @ S[:3, :3].T + S[:3, 3]


def umeyama(source, target):
    x, y = np.asarray(source), np.asarray(target)
    if len(x) < 3 or np.linalg.matrix_rank(x - x.mean(0), tol=1e-8) < 2:
        raise ValueError("Need three non-collinear centers")
    xc, yc = x - x.mean(0), y - y.mean(0)
    U, D, Vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1
    R = U @ np.diag(sign) @ Vt
    scale = (D @ sign) / np.mean(np.sum(xc * xc, axis=1))
    if scale <= 0:
        raise ValueError("Nonpositive scale")
    S = np.eye(4)
    S[:3, :3] = scale * R
    S[:3, 3] = y.mean(0) - scale * R @ x.mean(0)
    return S


def residual(S, x, y, weight):
    return np.linalg.norm((apply(S, x) - y) * weight, axis=1)


def matrix_from(S):
    scale = np.cbrt(np.linalg.det(S[:3, :3]))
    start = np.r_[np.log(scale), Rotation.from_matrix(S[:3, :3] / scale).as_rotvec(), S[:3, 3]]

    def matrix(v):
        m = np.eye(4)
        m[:3, :3] = np.exp(v[0]) * Rotation.from_rotvec(v[1:4]).as_matrix()
        m[:3, 3] = v[4:7]
        return m

    return matrix, start


def refine(S, x, y, weight, threshold):
    matrix, start = matrix_from(S)
    fit = least_squares(lambda v: ((apply(matrix(v), x) - y) * weight).ravel(), start,
                        loss="soft_l1", f_scale=max(threshold / 3, 0.01), max_nfev=200)
    return matrix(fit.x)


def robust(source, target, threshold, iterations=1000, z_weight=1., seed=7):
    x, y = np.asarray(source), np.asarray(target)
    rng = np.random.default_rng(seed)
    weight = np.array([1., 1., z_weight])
    best = np.zeros(len(x), dtype=bool)
    for _ in range(iterations):
        idx = rng.choice(len(x), 3, replace=False)
        try:
            S = umeyama(x[idx], y[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = residual(S, x, y, weight) < threshold
        if mask.sum() > best.sum():
            best = mask
    # A globally consistent set can have no dominant exact triplet; compare its
    # least-squares consensus and keep whichever explains more observations.
    try:
        S_all = umeyama(x, y)
        mask = residual(S_all, x, y, weight) < threshold
        if mask.sum() >= best.sum() and mask.sum() >= 3:
            best = mask
    except (ValueError, np.linalg.LinAlgError):
        pass
    if best.sum() < 3:
        raise ValueError("No nondegenerate similarity consensus")
    S = refine(umeyama(x[best], y[best]), x[best], y[best], weight, threshold)
    errors = residual(S, x, y, weight)
    return S, errors < threshold, errors
