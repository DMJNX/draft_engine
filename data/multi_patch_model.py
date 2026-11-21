import math
import numpy as np
from collections import defaultdict


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def fit_weighted_svd_prior(
    obs_matrix: np.ndarray, kish_matrix: np.ndarray, rank: int = 4, reg: float = 0.45, iters: int = 15
):
    """
    Scale-invariant Weighted ALS:
      - Pre-damps low-sample cells via n / (n + 50) so 1-game flukes cannot hijack factors.
      - Uses bounded weights w in [0, 1] with reg=0.45 so sparse early crawls shrink safely to 0
        while mature 50k+ matrices recover true latent archetype structure.
    """
    n_rows, n_cols = obs_matrix.shape
    if n_rows == 0 or n_cols == 0 or float(kish_matrix.sum()) < 300.0:
        return np.zeros_like(obs_matrix)

    rank = min(rank, n_rows, n_cols)
    weights = kish_matrix / (kish_matrix + 50.0)
    target = obs_matrix * weights

    rng = np.random.default_rng(42)
    U = rng.normal(0, 0.05, size=(n_rows, rank))
    V = rng.normal(0, 0.05, size=(n_cols, rank))
    I = np.eye(rank)

    for _ in range(iters):
        for i in range(n_rows):
            w = weights[i]
            if w.sum() < 0.25:
                U[i] = 0.0
                continue
            U[i] = np.linalg.solve((V.T * w) @ V + reg * I, (V.T * w) @ target[i])
        for j in range(n_cols):
            w = weights[:, j]
            if w.sum() < 0.25:
                V[j] = 0.0
                continue
            V[j] = np.linalg.solve((U.T * w) @ U + reg * I, (U.T * w) @ target[:, j])

    return np.clip(U @ V.T, -0.25, 0.25)


def build_decayed_residuals(
    patch_history: list[dict],
    rework_cutoffs: dict[int, int],
    decay_lambda: float = 0.87,
    k_pair: float = 250.0,
    is_enemy: bool = True,
):
    """Returns (final_deltas, kish_counts) for point estimates and LCB epistemic uncertainty."""
    num, den, den_sq = defaultdict(float), defaultdict(float), defaultdict(float)
    row_ids, col_ids = set(), set()

    for t, patch in enumerate(patch_history):
        w_t = decay_lambda ** t
        bw1 = patch.get("base_wr_1", patch.get("base_wr", {}))
        bw2 = patch.get("base_wr_2", patch.get("base_wr", {}))

        for (c1, c2), (wins, n) in patch["pairs"].items():
            if n <= 0 or c1 not in bw1 or c2 not in bw2:
                continue
            if t > rework_cutoffs.get(c1, 999) or t > rework_cutoffs.get(c2, 999):
                continue

            obs_p = (wins + 1.0) / (n + 2.0)
            exp_logit = (
                logit(bw1[c1]) - logit(bw2[c2])
                if is_enemy
                else logit(bw1[c1]) + logit(bw2[c2])
            )
            delta_t = logit(obs_p) - exp_logit
            num[(c1, c2)] += w_t * n * delta_t
            den[(c1, c2)] += w_t * n
            den_sq[(c1, c2)] += (w_t ** 2) * n
            row_ids.add(c1)
            col_ids.add(c2)

    rows, cols = sorted(row_ids), sorted(col_ids)
    if not rows or not cols:
        return {}, {}

    r_idx = {c: i for i, c in enumerate(rows)}
    c_idx = {c: j for j, c in enumerate(cols)}
    obs_mat = np.zeros((len(rows), len(cols)))
    kish_mat = np.zeros((len(rows), len(cols)))

    for (c1, c2), total_w in den.items():
        i, j = r_idx[c1], c_idx[c2]
        obs_mat[i, j] = num[(c1, c2)] / total_w
        kish_mat[i, j] = (total_w ** 2) / den_sq[(c1, c2)]

    prior_mat = fit_weighted_svd_prior(obs_mat, kish_mat, rank=4, reg=0.45)

    final_deltas, kish_counts = {}, {}
    for i, c1 in enumerate(rows):
        for j, c2 in enumerate(cols):
            n_eff = float(kish_mat[i, j])
            alpha = n_eff / (n_eff + k_pair)
            final_deltas[(c1, c2)] = float(alpha * obs_mat[i, j] + (1.0 - alpha) * prior_mat[i, j])
            kish_counts[(c1, c2)] = n_eff

    return final_deltas, kish_counts