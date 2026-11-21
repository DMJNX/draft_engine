import json, math, os, pickle, sqlite3, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from core.scoring_engine import ROLES, MATCHUP_HEATMAP, SYNERGY_HEATMAP
from data.db_loader import load_engine_from_db
from ml.stacker_runtime import build_wide_pair_indices, extract_neural_block, extract_tabular_block
from ml.transformer_model import MECH_DIM, DraftEnsembleONNX, DraftSetTransformer

ROLE_TO_IDX = {"TOP": 0, "JGL": 1, "MID": 2, "BOT": 3, "SUP": 4}
ELO_TO_IDX = {"IRON": 0, "BRONZE": 0, "SILVER": 1, "GOLD": 1, "PLATINUM": 2, "EMERALD": 3, "DIAMOND": 4, "MASTER": 5}


def _patch_key(p: str):
    return tuple(int(x) for x in p.split(".") if x.isdigit())


def _per_sample_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    ll = float(np.mean(_per_sample_ll(y, pc)))
    brier = float(np.mean((pc - y) ** 2))
    acc = 100.0 * float(np.mean((pc >= 0.5) == y))
    order = np.argsort(pc)
    ys = y[order]
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, len(y) - n_pos)
    auc = float((np.arange(1, len(y) + 1)[ys == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        m = (pc >= bins[i]) & (pc < bins[i + 1]) if i < 9 else (pc >= bins[i]) & (pc <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(pc)) * abs(float(pc[m].mean()) - float(y[m].mean()))
    return {"Acc": acc, "LogLoss": ll, "Brier": brier, "AUC": auc, "ECE": float(ece)}


def _print_sig(label: str, y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray):
    diff = _per_sample_ll(y, p_old) - _per_sample_ll(y, p_new)
    gain = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(max(1, len(diff))))
    z = gain / max(se, 1e-9)
    verdict = "SIGNIFICANT" if z >= 1.96 else ("WORSE" if z <= -1.96 else "inconclusive")
    print(f"  ΔLogLoss ({label:<36}): {gain:+.5f}  (z={z:+.2f} -> {verdict})")


def _print_table(title: str, rows: list[tuple[str, dict]]):
    print(f"\n=== {title} ===")
    print(f"{'Model / Benchmark':<46} | {'Acc%':>6} | {'LogLoss':>7} | {'Brier':>6} | {'AUC':>6} | {'ECE':>6}")
    print("-" * 90)
    for name, m in rows:
        print(f"{name:<46} | {m['Acc']:>6.2f} | {m['LogLoss']:>7.4f} | {m['Brier']:>6.4f} | {m['AUC']:>6.4f} | {m['ECE']:>6.4f}")
    print("-" * 90)
    print(f"{'Published Ref: DraftGap (Apr 2026)':<46} | {'54.66':>6} | {'0.6869':>7} | {'0.2469':>6} | {'~0.56':>6} | {'~0.010':>6}")
    print(f"{'Published Ref: LoLDraftAI (2026, ~3M)':<46} | {'55.90':>6} | {'0.6824':>7} | {'0.2447':>6} | {'~0.58':>6} | {'~0.009':>6}")


def _fit_temperature(logits: np.ndarray, y: np.ndarray, sw: np.ndarray) -> float:
    best_t, best_ll = 1.0, 1e9
    for t in np.linspace(0.75, 1.50, 76):
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits / t, -15.0, 15.0))), 1e-6, 1.0 - 1e-6)
        ll = float(np.sum(sw * -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) / np.sum(sw))
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    return best_t


def _fit_joint_bt_10hot(b_rc, r_rc, y, sw, D, l2=350.0):
    bf, rf = b_rc.ravel(), r_rc.ravel()
    theta = np.zeros(D + 1, dtype=np.float64)
    for _ in range(5):
        w, b = theta[:D], theta[D]
        z = np.clip(w[b_rc].sum(axis=1) - w[r_rc].sum(axis=1) + b, -15.0, 15.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = sw * (p - y)
        W = np.clip(p * (1.0 - p), 1e-6, 0.25) * sw
        err5 = np.repeat(err, 5)
        grad = np.empty(D + 1, dtype=np.float64)
        grad[:D] = np.bincount(bf, weights=err5, minlength=D) - np.bincount(rf, weights=err5, minlength=D) + l2 * w
        grad[0] = 0.0
        grad[D] = float(err.sum()) + 1e-4 * b

        def matvec(v):
            vw, vb = v[:D], v[D]
            xv = vw[b_rc].sum(axis=1) - vw[r_rc].sum(axis=1) + vb
            wxv = W * xv
            wxv5 = np.repeat(wxv, 5)
            out = np.empty(D + 1, dtype=np.float64)
            out[:D] = np.bincount(bf, weights=wxv5, minlength=D) - np.bincount(rf, weights=wxv5, minlength=D) + l2 * vw
            out[0] = vw[0]
            out[D] = float(wxv.sum()) + 1e-4 * vb
            return out

        step = np.zeros(D + 1, dtype=np.float64)
        rv = -grad
        pd = rv.copy()
        rs = float(rv @ rv)
        for _ in range(18):
            if rs < 1e-10:
                break
            Ap = matvec(pd)
            den = float(pd @ Ap)
            if den <= 1e-12:
                break
            a = rs / den
            step += a * pd
            rv -= a * Ap
            rs_new = float(rv @ rv)
            if rs_new < 1e-10:
                break
            pd = rv + (rs_new / rs) * pd
            rs = rs_new
        theta += np.clip(step, -1.0, 1.0)
        theta[0] = 0.0
    return theta[:D]


def _fit_joint_ridge_10hot(b_rc, r_rc, target, sw, D, l2=220.0):
    bf, rf = b_rc.ravel(), r_rc.ravel()
    st = sw * target
    st5 = np.repeat(st, 5)
    rhs = np.empty(D + 1, dtype=np.float64)
    rhs[:D] = np.bincount(bf, weights=st5, minlength=D) - np.bincount(rf, weights=st5, minlength=D)
    rhs[0] = 0.0
    rhs[D] = float(st.sum())

    def matvec(v):
        vw, vb = v[:D], v[D]
        xv = vw[b_rc].sum(axis=1) - vw[r_rc].sum(axis=1) + vb
        wxv = sw * xv
        wxv5 = np.repeat(wxv, 5)
        out = np.empty(D + 1, dtype=np.float64)
        out[:D] = np.bincount(bf, weights=wxv5, minlength=D) - np.bincount(rf, weights=wxv5, minlength=D) + l2 * vw
        out[0] = vw[0]
        out[D] = float(wxv.sum()) + 1e-4 * vb
        return out

    x = np.zeros(D + 1, dtype=np.float64)
    rv = rhs.copy()
    pd = rv.copy()
    rs = float(rv @ rv)
    for _ in range(22):
        if rs < 1e-10:
            break
        Ap = matvec(pd)
        den = float(pd @ Ap)
        if den <= 1e-12:
            break
        a = rs / den
        x += a * pd
        rv -= a * Ap
        rs_new = float(rv @ rv)
        if rs_new < 1e-10:
            break
        pd = rv + (rs_new / rs) * pd
        rs = rs_new
    x[0] = 0.0
    return x[:D]


def _fit_wide_pair_logreg(idx_mat, sgn_mat, y, sw, D, num_rc, l2_champ=350.0, l2_pair=150.0):
    _, K = idx_mat.shape
    flat_idx = idx_mat.ravel()
    flat_sgn = sgn_mat.ravel()
    reg = np.full(D + 1, l2_pair, dtype=np.float64)
    reg[:num_rc] = l2_champ
    reg[0] = 1.0
    reg[D] = 1e-4

    theta = np.zeros(D + 1, dtype=np.float64)
    for _ in range(5):
        w, b = theta[:D], theta[D]
        z = np.clip((w[idx_mat] * sgn_mat).sum(axis=1) + b, -15.0, 15.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = sw * (p - y)
        W = np.clip(p * (1.0 - p), 1e-6, 0.25) * sw
        grad = np.empty(D + 1, dtype=np.float64)
        grad[:D] = np.bincount(flat_idx, weights=np.repeat(err, K) * flat_sgn, minlength=D) + reg[:D] * w
        grad[0] = 0.0
        grad[D] = float(err.sum()) + reg[D] * b

        def matvec(v):
            vw, vb = v[:D], v[D]
            xv = (vw[idx_mat] * sgn_mat).sum(axis=1) + vb
            wxv = W * xv
            out = np.empty(D + 1, dtype=np.float64)
            out[:D] = np.bincount(flat_idx, weights=np.repeat(wxv, K) * flat_sgn, minlength=D) + reg[:D] * vw
            out[0] = vw[0]
            out[D] = float(wxv.sum()) + reg[D] * vb
            return out

        step = np.zeros(D + 1, dtype=np.float64)
        rv = -grad
        pd = rv.copy()
        rs = float(rv @ rv)
        for _ in range(20):
            if rs < 1e-10:
                break
            Ap = matvec(pd)
            den = float(pd @ Ap)
            if den <= 1e-12:
                break
            a = rs / den
            step += a * pd
            rv -= a * Ap
            rs_new = float(rv @ rv)
            if rs_new < 1e-10:
                break
            pd = rv + (rs_new / rs) * pd
            rs = rs_new
        theta += np.clip(step, -1.0, 1.0)
        theta[0] = 0.0
    return theta


def _fit_wide_pair_ridge(idx_mat, sgn_mat, target, sw, D, num_rc, l2_champ=220.0, l2_pair=65.0):
    _, K = idx_mat.shape
    flat_idx = idx_mat.ravel()
    flat_sgn = sgn_mat.ravel()
    reg = np.full(D + 1, l2_pair, dtype=np.float64)
    reg[:num_rc] = l2_champ
    reg[0] = 1.0
    reg[D] = 1e-4

    st = sw * target
    rhs = np.empty(D + 1, dtype=np.float64)
    rhs[:D] = np.bincount(flat_idx, weights=np.repeat(st, K) * flat_sgn, minlength=D)
    rhs[0] = 0.0
    rhs[D] = float(st.sum())

    def matvec(v):
        vw, vb = v[:D], v[D]
        xv = (vw[idx_mat] * sgn_mat).sum(axis=1) + vb
        wxv = sw * xv
        out = np.empty(D + 1, dtype=np.float64)
        out[:D] = np.bincount(flat_idx, weights=np.repeat(wxv, K) * flat_sgn, minlength=D) + reg[:D] * vw
        out[0] = vw[0]
        out[D] = float(wxv.sum()) + reg[D] * vb
        return out

    x = np.zeros(D + 1, dtype=np.float64)
    rv = rhs.copy()
    pd = rv.copy()
    rs = float(rv @ rv)
    for _ in range(24):
        if rs < 1e-10:
            break
        Ap = matvec(pd)
        den = float(pd @ Ap)
        if den <= 1e-12:
            break
        a = rs / den
        x += a * pd
        rv -= a * Ap
        rs_new = float(rv @ rv)
        if rs_new < 1e-10:
            break
        pd = rv + (rs_new / rs) * pd
        rs = rs_new
    x[0] = 0.0
    return x


def _soft_impute_matrix(sum_mat, cnt_mat, tau, rank=10, skew=True, n_iter=6):
    if sum_mat.shape[0] < 4:
        return np.zeros_like(sum_mat)
    W = cnt_mat / (cnt_mat + tau)
    Y = np.divide(sum_mat, np.maximum(cnt_mat, 1e-6))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Wt = torch.from_numpy(W.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    Zt = torch.zeros_like(Yt)
    k = min(rank, sum_mat.shape[0] - 1)
    with torch.no_grad():
        for _ in range(n_iter):
            T = Wt * Yt + (1.0 - Wt) * Zt
            T = 0.5 * (T - T.T) if skew else 0.5 * (T + T.T)
            U, S, Vh = torch.linalg.svd(T, full_matrices=False)
            lam = S[k].item() if len(S) > k else 0.0
            S_shrunk = torch.clamp(S[:k] - 0.5 * lam, min=0.0)
            Zt = (U[:, :k] * S_shrunk.unsqueeze(0)) @ Vh[:k, :]
            Zt = 0.5 * (Zt - Zt.T) if skew else 0.5 * (Zt + Zt.T)
    return Zt.cpu().numpy().astype(np.float64)


def _fit_dense_logreg(X, y, l2=15.0):
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    reg = np.full(Xb.shape[1], l2, dtype=np.float64)
    reg[0] = 1e-4
    for _ in range(8):
        z = np.clip(Xb @ w, -15.0, 15.0)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(p * (1.0 - p), 1e-6, 0.25)
        grad = Xb.T @ (p - y) + reg * w
        H = (Xb.T * W) @ Xb
        H[np.diag_indices_from(H)] += reg
        w -= np.linalg.solve(H, grad)
    return w


def _predict_dense_logreg(X, w):
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -15.0, 15.0)))


def _fit_fold_stats(b_rc, r_rc, y_win, b_gd, r_gd, b_xpd, r_xpd, dur_min, obj_arr, sw, sw_long, sw_fast, num_rc):
    w_jbt_long = _fit_joint_bt_10hot(b_rc, r_rc, y_win, sw_long, num_rc, l2=350.0)
    w_jbt_med = _fit_joint_bt_10hot(b_rc, r_rc, y_win, sw, num_rc, l2=350.0)
    w_jbt_fast = _fit_joint_bt_10hot(b_rc, r_rc, y_win, sw_fast, num_rc, l2=420.0)

    gd_diff = (b_gd - r_gd) / 1500.0
    xpd_diff = (b_xpd - r_xpd) / 1200.0
    team_gd = gd_diff.sum(axis=1)
    team_xpd = xpd_diff.sum(axis=1)
    ft_y = obj_arr[:, 0] - np.mean(obj_arr[:, 0])
    grb_y = obj_arr[:, 1]

    w_gd = _fit_joint_ridge_10hot(b_rc, r_rc, team_gd, sw, num_rc, l2=220.0)
    w_xpd = _fit_joint_ridge_10hot(b_rc, r_rc, team_xpd, sw, num_rc, l2=220.0)
    w_ft = _fit_joint_ridge_10hot(b_rc, r_rc, ft_y, sw, num_rc, l2=250.0)
    w_grb = _fit_joint_ridge_10hot(b_rc, r_rc, grb_y, sw, num_rc, l2=250.0)

    idx_w, sgn_w, pair_vocab, D_wide = build_wide_pair_indices(b_rc, r_rc, num_rc, pair_vocab=None, min_cnt=25)
    theta_wide_win = _fit_wide_pair_logreg(idx_w, sgn_w, y_win, sw, D_wide, num_rc, l2_champ=350.0, l2_pair=150.0)
    theta_wide_gd = _fit_wide_pair_ridge(idx_w, sgn_w, team_gd, sw, D_wide, num_rc, l2_champ=220.0, l2_pair=65.0)
    pair_vk = np.array(sorted(pair_vocab.keys()), dtype=np.int64) if pair_vocab else np.empty(0, dtype=np.int64)
    pair_vv = np.array([pair_vocab[int(k)] for k in pair_vk], dtype=np.int64) if pair_vocab else np.empty(0, dtype=np.int64)

    rc_gd_s = np.zeros(num_rc, dtype=np.float64)
    rc_xpd_s = np.zeros(num_rc, dtype=np.float64)
    rc_cnt = np.zeros(num_rc, dtype=np.float64)
    rc_early_w = np.zeros(num_rc, dtype=np.float64)
    rc_early_n = np.zeros(num_rc, dtype=np.float64)
    rc_late_w = np.zeros(num_rc, dtype=np.float64)
    rc_late_n = np.zeros(num_rc, dtype=np.float64)

    is_early = (dur_min < 27.0).astype(np.float64)
    is_late = (dur_min >= 32.0).astype(np.float64)
    sw5 = np.repeat(sw, 5)
    y5 = np.repeat(y_win, 5)
    e5 = np.repeat(is_early * sw, 5)
    l5 = np.repeat(is_late * sw, 5)

    bf, rf = b_rc.ravel(), r_rc.ravel()
    rc_cnt += np.bincount(bf, weights=sw5, minlength=num_rc) + np.bincount(rf, weights=sw5, minlength=num_rc)
    rc_gd_s += np.bincount(bf, weights=sw5 * gd_diff.ravel(), minlength=num_rc) - np.bincount(rf, weights=sw5 * gd_diff.ravel(), minlength=num_rc)
    rc_xpd_s += np.bincount(bf, weights=sw5 * xpd_diff.ravel(), minlength=num_rc) - np.bincount(rf, weights=sw5 * xpd_diff.ravel(), minlength=num_rc)
    rc_early_n += np.bincount(bf, weights=e5, minlength=num_rc) + np.bincount(rf, weights=e5, minlength=num_rc)
    rc_early_w += np.bincount(bf, weights=e5 * y5, minlength=num_rc) + np.bincount(rf, weights=e5 * (1.0 - y5), minlength=num_rc)
    rc_late_n += np.bincount(bf, weights=l5, minlength=num_rc) + np.bincount(rf, weights=l5, minlength=num_rc)
    rc_late_w += np.bincount(bf, weights=l5 * y5, minlength=num_rc) + np.bincount(rf, weights=l5 * (1.0 - y5), minlength=num_rc)

    base_gd = rc_gd_s / (rc_cnt + 40.0)
    base_xpd = rc_xpd_s / (rc_cnt + 40.0)
    scale_slope = (rc_late_w + 20.0) / (rc_late_n + 40.0) - (rc_early_w + 20.0) / (rc_early_n + 40.0)
    base_gd[0] = base_xpd[0] = scale_slope[0] = 0.0

    lane_gd_res = gd_diff - (base_gd[b_rc] - base_gd[r_rc])
    lane_xpd_res = xpd_diff - (base_xpd[b_rc] - base_xpd[r_rc])
    team_gd_res = team_gd - (w_gd[b_rc].sum(axis=1) - w_gd[r_rc].sum(axis=1))
    team_xpd_res = team_xpd - (w_xpd[b_rc].sum(axis=1) - w_xpd[r_rc].sum(axis=1))
    win_exp = 1.0 / (1.0 + np.exp(-np.clip(w_jbt_med[b_rc].sum(axis=1) - w_jbt_med[r_rc].sum(axis=1), -8.0, 8.0)))
    win_res = y_win - win_exp
    ft_res = ft_y - (w_ft[b_rc].sum(axis=1) - w_ft[r_rc].sum(axis=1))
    grb_res = grb_y - (w_grb[b_rc].sum(axis=1) - w_grb[r_rc].sum(axis=1))

    active_rcs = np.nonzero(rc_cnt >= 20.0)[0]
    active_rcs = active_rcs[active_rcs > 0]
    rc_to_comp = np.full(num_rc, -1, dtype=np.int64)
    rc_to_comp[active_rcs] = np.arange(len(active_rcs), dtype=np.int64)
    K_act = len(active_rcs)

    lane_gd_map, lane_xpd_map, lane_win_map, lane_obj_map = {}, {}, {}, {}
    role_svd_gd, role_svd_xpd, role_svd_win, role_comp_maps = [], [], [], []

    for ri in range(5):
        cb, cr = b_rc[:, ri], r_rc[:, ri]
        vp = (cb > 0) & (cr > 0)
        cb_v, cr_v = cb[vp], cr[vp]
        k1, k2 = cb_v * num_rc + cr_v, cr_v * num_rc + cb_v
        keys = np.concatenate([k1, k2])
        w_rep = np.concatenate([sw[vp], sw[vp]])
        gd_rep = np.concatenate([lane_gd_res[vp, ri], -lane_gd_res[vp, ri]])
        xp_rep = np.concatenate([lane_xpd_res[vp, ri], -lane_xpd_res[vp, ri]])
        wn_rep = np.concatenate([win_res[vp], -win_res[vp]])
        ob_rep = np.concatenate([0.5 * (ft_res[vp] + grb_res[vp]), -0.5 * (ft_res[vp] + grb_res[vp])])

        uk, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv, weights=w_rep)
        s_gd = np.bincount(inv, weights=w_rep * gd_rep) / (cnt + 18.0)
        s_xp = np.bincount(inv, weights=w_rep * xp_rep) / (cnt + 18.0)
        s_wn = np.bincount(inv, weights=w_rep * wn_rep) / (cnt + 75.0)
        s_ob = np.bincount(inv, weights=w_rep * ob_rep) / (cnt + 35.0)
        for u_i, kv in enumerate(uk):
            if cnt[u_i] >= 2.0:
                ik = int(kv)
                lane_gd_map[ik] = float(s_gd[u_i])
                lane_xpd_map[ik] = float(s_xp[u_i])
                lane_win_map[ik] = float(s_wn[u_i])
                lane_obj_map[ik] = float(s_ob[u_i])

        role_rcs = np.unique(np.concatenate([cb_v, cr_v]))
        role_rcs = role_rcs[(role_rcs > 0) & (rc_cnt[role_rcs] >= 20.0)]
        rmap = np.full(num_rc, -1, dtype=np.int64)
        rmap[role_rcs] = np.arange(len(role_rcs), dtype=np.int64)
        Kr = len(role_rcs)
        c1, c2 = rmap[cb], rmap[cr]
        ok = (c1 >= 0) & (c2 >= 0)
        sm_gd = np.zeros((Kr, Kr), dtype=np.float64)
        sm_xp = np.zeros((Kr, Kr), dtype=np.float64)
        sm_wn = np.zeros((Kr, Kr), dtype=np.float64)
        cm = np.zeros((Kr, Kr), dtype=np.float64)
        if ok.any():
            i1, i2, ww = c1[ok], c2[ok], sw[ok]
            np.add.at(cm, (i1, i2), ww); np.add.at(cm, (i2, i1), ww)
            np.add.at(sm_gd, (i1, i2), ww * lane_gd_res[ok, ri]); np.add.at(sm_gd, (i2, i1), -ww * lane_gd_res[ok, ri])
            np.add.at(sm_xp, (i1, i2), ww * lane_xpd_res[ok, ri]); np.add.at(sm_xp, (i2, i1), -ww * lane_xpd_res[ok, ri])
            np.add.at(sm_wn, (i1, i2), ww * win_res[ok]); np.add.at(sm_wn, (i2, i1), -ww * win_res[ok])
        role_svd_gd.append(_soft_impute_matrix(sm_gd, cm, tau=20.0, rank=8, skew=True))
        role_svd_xpd.append(_soft_impute_matrix(sm_xp, cm, tau=20.0, rank=8, skew=True))
        role_svd_win.append(_soft_impute_matrix(sm_wn, cm, tau=80.0, rank=6, skew=True))
        role_comp_maps.append(rmap)

    hm_c = np.array([[MATCHUP_HEATMAP[r1][r2] for r2 in ROLES] for r1 in ROLES], dtype=np.float64)
    hm_s = np.array([[SYNERGY_HEATMAP.get(r1, {}).get(r2, 0.0) for r2 in ROLES] for r1 in ROLES], dtype=np.float64)

    cross_win_map, cross_gd_map, cross_xpd_map, cross_lgd_map = {}, {}, {}, {}
    g_ctr_gd = np.zeros((K_act, K_act), dtype=np.float64)
    g_ctr_wn = np.zeros((K_act, K_act), dtype=np.float64)
    g_ctr_cn = np.zeros((K_act, K_act), dtype=np.float64)

    for r1 in range(5):
        for r2 in range(5):
            if r1 == r2 or hm_c[r1, r2] < 0.20:
                continue
            cb, cr = b_rc[:, r1], r_rc[:, r2]
            vp1 = (cb > 0) & (cr > 0)
            vp2 = (r_rc[:, r1] > 0) & (b_rc[:, r2] > 0)
            k1 = cb[vp1] * num_rc + cr[vp1]
            k2 = r_rc[vp2, r1] * num_rc + b_rc[vp2, r2]
            keys = np.concatenate([k1, k2])
            w_rep = np.concatenate([sw[vp1], sw[vp2]])
            wn_rep = np.concatenate([win_res[vp1], -win_res[vp2]])
            tgd_rep = np.concatenate([team_gd_res[vp1], -team_gd_res[vp2]])
            txp_rep = np.concatenate([team_xpd_res[vp1], -team_xpd_res[vp2]])
            lgd_rep = np.concatenate([lane_gd_res[vp1, r1], -lane_gd_res[vp2, r1]])

            uk, inv = np.unique(keys, return_inverse=True)
            cnt = np.bincount(inv, weights=w_rep)
            vw = np.bincount(inv, weights=w_rep * wn_rep) / (cnt + 110.0)
            vg = np.bincount(inv, weights=w_rep * tgd_rep) / (cnt + 35.0)
            vx = np.bincount(inv, weights=w_rep * txp_rep) / (cnt + 35.0)
            vl = np.bincount(inv, weights=w_rep * lgd_rep) / (cnt + 25.0)
            for u_i, kv in enumerate(uk):
                if cnt[u_i] >= 3.0:
                    ik = int(kv)
                    cross_win_map[ik] = float(vw[u_i])
                    cross_gd_map[ik] = float(vg[u_i])
                    cross_xpd_map[ik] = float(vx[u_i])
                    cross_lgd_map[ik] = float(vl[u_i])

            i1, i2 = rc_to_comp[cb], rc_to_comp[cr]
            ok = (i1 >= 0) & (i2 >= 0)
            if ok.any():
                ww = sw[ok] * hm_c[r1, r2]
                np.add.at(g_ctr_cn, (i1[ok], i2[ok]), ww); np.add.at(g_ctr_cn, (i2[ok], i1[ok]), ww)
                np.add.at(g_ctr_gd, (i1[ok], i2[ok]), ww * team_gd_res[ok]); np.add.at(g_ctr_gd, (i2[ok], i1[ok]), -ww * team_gd_res[ok])
                np.add.at(g_ctr_wn, (i1[ok], i2[ok]), ww * win_res[ok]); np.add.at(g_ctr_wn, (i2[ok], i1[ok]), -ww * win_res[ok])

    svd_cross_gd = _soft_impute_matrix(g_ctr_gd, g_ctr_cn, tau=30.0, rank=12, skew=True)
    svd_cross_wn = _soft_impute_matrix(g_ctr_wn, g_ctr_cn, tau=100.0, rank=10, skew=True)

    syn_win_map, syn_gd_map, syn_xpd_map, syn_lgd_map, syn_obj_map = {}, {}, {}, {}, {}
    g_syn_gd = np.zeros((K_act, K_act), dtype=np.float64)
    g_syn_wn = np.zeros((K_act, K_act), dtype=np.float64)
    g_syn_cn = np.zeros((K_act, K_act), dtype=np.float64)

    for r1 in range(5):
        for r2 in range(r1 + 1, 5):
            if hm_s[r1, r2] < 0.20:
                continue
            vpb = (b_rc[:, r1] > 0) & (b_rc[:, r2] > 0)
            vpr = (r_rc[:, r1] > 0) & (r_rc[:, r2] > 0)
            kb = b_rc[vpb, r1] * num_rc + b_rc[vpb, r2]
            kr = r_rc[vpr, r1] * num_rc + r_rc[vpr, r2]
            keys = np.concatenate([kb, kr])
            w_rep = np.concatenate([sw[vpb], sw[vpr]])
            wn_rep = np.concatenate([win_res[vpb], -win_res[vpr]])
            tgd_rep = np.concatenate([team_gd_res[vpb], -team_gd_res[vpr]])
            txp_rep = np.concatenate([team_xpd_res[vpb], -team_xpd_res[vpr]])
            pair_lgd = 0.5 * (lane_gd_res[:, r1] + lane_gd_res[:, r2])
            lgd_rep = np.concatenate([pair_lgd[vpb], -pair_lgd[vpr]])
            ob_rep = np.concatenate([0.5 * (ft_res[vpb] + grb_res[vpb]), -0.5 * (ft_res[vpr] + grb_res[vpr])])

            uk, inv = np.unique(keys, return_inverse=True)
            cnt = np.bincount(inv, weights=w_rep)
            vw = np.bincount(inv, weights=w_rep * wn_rep) / (cnt + 110.0)
            vg = np.bincount(inv, weights=w_rep * tgd_rep) / (cnt + 35.0)
            vx = np.bincount(inv, weights=w_rep * txp_rep) / (cnt + 35.0)
            vl = np.bincount(inv, weights=w_rep * lgd_rep) / (cnt + 25.0)
            vo = np.bincount(inv, weights=w_rep * ob_rep) / (cnt + 45.0)
            for u_i, kv in enumerate(uk):
                if cnt[u_i] >= 3.0:
                    ik = int(kv)
                    syn_win_map[ik] = float(vw[u_i])
                    syn_gd_map[ik] = float(vg[u_i])
                    syn_xpd_map[ik] = float(vx[u_i])
                    syn_lgd_map[ik] = float(vl[u_i])
                    syn_obj_map[ik] = float(vo[u_i])

            for team_rc, sgn_t in ((b_rc, 1.0), (r_rc, -1.0)):
                i1, i2 = rc_to_comp[team_rc[:, r1]], rc_to_comp[team_rc[:, r2]]
                ok = (i1 >= 0) & (i2 >= 0)
                if ok.any():
                    ww = sw[ok] * hm_s[r1, r2]
                    np.add.at(g_syn_cn, (i1[ok], i2[ok]), ww); np.add.at(g_syn_cn, (i2[ok], i1[ok]), ww)
                    np.add.at(g_syn_gd, (i1[ok], i2[ok]), ww * sgn_t * team_gd_res[ok]); np.add.at(g_syn_gd, (i2[ok], i1[ok]), ww * sgn_t * team_gd_res[ok])
                    np.add.at(g_syn_wn, (i1[ok], i2[ok]), ww * sgn_t * win_res[ok]); np.add.at(g_syn_wn, (i2[ok], i1[ok]), ww * sgn_t * win_res[ok])

    svd_syn_gd = _soft_impute_matrix(g_syn_gd, g_syn_cn, tau=30.0, rank=12, skew=False)
    svd_syn_wn = _soft_impute_matrix(g_syn_wn, g_syn_cn, tau=100.0, rank=10, skew=False)

    return {
        "w_jbt_long": w_jbt_long, "w_jbt_med": w_jbt_med, "w_jbt_fast": w_jbt_fast,
        "w_gd": w_gd, "w_xpd": w_xpd, "w_ft": w_ft, "w_grb": w_grb,
        "pair_vk": pair_vk, "pair_vv": pair_vv, "theta_wide_win": theta_wide_win, "theta_wide_gd": theta_wide_gd,
        "base_gd": base_gd, "base_xpd": base_xpd, "scale_slope": scale_slope,
        "lane_gd_map": lane_gd_map, "lane_xpd_map": lane_xpd_map, "lane_win_map": lane_win_map, "lane_obj_map": lane_obj_map,
        "role_svd_gd": role_svd_gd, "role_svd_xpd": role_svd_xpd, "role_svd_win": role_svd_win, "role_comp_maps": role_comp_maps,
        "cross_win_map": cross_win_map, "cross_gd_map": cross_gd_map, "cross_xpd_map": cross_xpd_map, "cross_lgd_map": cross_lgd_map,
        "syn_win_map": syn_win_map, "syn_gd_map": syn_gd_map, "syn_xpd_map": syn_xpd_map, "syn_lgd_map": syn_lgd_map, "syn_obj_map": syn_obj_map,
        "rc_to_comp": rc_to_comp, "svd_cross_gd": svd_cross_gd, "svd_cross_wn": svd_cross_wn,
        "svd_syn_gd": svd_syn_gd, "svd_syn_wn": svd_syn_wn,
        "hm_c": hm_c, "hm_s": hm_s, "num_rc": num_rc,
    }


def _run_v4_on_test(test_c, test_mv, test_elo, ckpt_path="data/draft_transformer_ensemble.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    models = [
        DraftSetTransformer(
            num_champs=ckpt["num_champs"], num_patches=ckpt["num_patches"],
            d_model=96, nhead=4, num_layers=3, fm_dim=24, dropout=0.22,
        )
        for _ in range(len(ckpt["temps"]))
    ]
    ens = DraftEnsembleONNX(models, temps=ckpt["temps"])
    ens.load_state_dict(ckpt["state_dict"])
    ens.to(device).eval()

    n = len(test_c)
    inv_t = [1.0 / max(0.25, t) for t in ckpt["temps"]]
    r_base = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long, device=device)
    t_base = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long, device=device)

    out_w = np.zeros(n, dtype=np.float64)
    out_dw = np.zeros((n, 4), dtype=np.float64)
    out_dc = np.zeros((n, 4), dtype=np.float64)
    out_tm = np.zeros((n, 5, 2), dtype=np.float64)
    out_ob = np.zeros((n, 4), dtype=np.float64)

    bs = 512
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(n, i + bs)
            c = torch.from_numpy(test_c[i:j]).to(device)
            mv = torch.from_numpy(test_mv[i:j]).to(device)
            elo = torch.from_numpy(test_elo[i:j]).to(device)
            b = j - i
            r = r_base.unsqueeze(0).expand(b, -1)
            t = t_base.unsqueeze(0).expand(b, -1)
            s = torch.zeros_like(c)
            l1 = torch.zeros(b, dtype=torch.float32, device=device)

            w_acc = dw_acc = dc_acc = tm_acc = ob_acc = 0.0
            for m_idx, m in enumerate(ens.models):
                w, dw, tm, dc, ob, _ = m(c, mv, r, t, s, elo, l1)
                w_acc = w_acc + (w.cpu().numpy() * inv_t[m_idx])
                dw_acc = dw_acc + dw.cpu().numpy()
                dc_acc = dc_acc + torch.softmax(dc, dim=-1).cpu().numpy()
                tm_acc = tm_acc + tm.cpu().numpy()
                ob_acc = ob_acc + ob.cpu().numpy()

            k_m = float(len(ens.models))
            out_w[i:j] = w_acc / k_m
            out_dw[i:j] = dw_acc / k_m
            out_dc[i:j] = dc_acc / k_m
            out_tm[i:j] = tm_acc / k_m
            out_ob[i:j] = ob_acc / k_m
    return out_w, out_dw, out_dc, out_tm, out_ob


def _train_antisym_lgb(anti_tr, sym_tr, y_tr, sw_tr, fold_ids, anti_te, sym_te, feat_names, param_sets):
    n_tr = len(y_tr)
    has_te = anti_te is not None
    oof_z = np.zeros(n_tr, dtype=np.float64)
    te_z = np.zeros(len(anti_te), dtype=np.float64) if has_te else None
    imp = np.zeros(len(feat_names), dtype=np.float64)
    booster_strs = []

    def _pack(a, s, side_val):
        return np.column_stack([a, s, np.full(len(a), side_val, dtype=np.float64)])

    if has_te:
        X_te_orig = _pack(anti_te, sym_te, +1.0)
        X_te_mirr = _pack(-anti_te, sym_te, -1.0)
    n_cfg = len(param_sets)

    for c_idx, base_p in enumerate(param_sets):
        c_oof = np.zeros(n_tr, dtype=np.float64)
        c_te = np.zeros(len(anti_te), dtype=np.float64) if has_te else None
        for f in range(5):
            vm = (fold_ids == f)
            tm = ~vm
            X_tr_aug = np.vstack([_pack(anti_tr[tm], sym_tr[tm], +1.0), _pack(-anti_tr[tm], sym_tr[tm], -1.0)])
            y_tr_aug = np.concatenate([y_tr[tm], 1.0 - y_tr[tm]])
            sw_tr_aug = np.concatenate([sw_tr[tm], sw_tr[tm]])

            X_va_orig = _pack(anti_tr[vm], sym_tr[vm], +1.0)
            X_va_mirr = _pack(-anti_tr[vm], sym_tr[vm], -1.0)
            X_va_aug = np.vstack([X_va_orig, X_va_mirr])
            y_va_aug = np.concatenate([y_tr[vm], 1.0 - y_tr[vm]])
            sw_va_aug = np.concatenate([sw_tr[vm], sw_tr[vm]])

            dtr = lgb.Dataset(X_tr_aug, label=y_tr_aug, weight=sw_tr_aug, feature_name=feat_names)
            dva = lgb.Dataset(X_va_aug, label=y_va_aug, weight=sw_va_aug, feature_name=feat_names, reference=dtr)

            p_fold = dict(base_p, seed=600 + 31 * c_idx + f)
            bst = lgb.train(
                p_fold, dtr, num_boost_round=1600, valid_sets=[dva],
                callbacks=[lgb.early_stopping(70, verbose=False)],
            )
            booster_strs.append(bst.model_to_string())
            imp += bst.feature_importance(importance_type="gain") / (5.0 * n_cfg)
            c_oof[vm] = 0.5 * (bst.predict(X_va_orig, raw_score=True) - bst.predict(X_va_mirr, raw_score=True))
            if has_te:
                c_te += 0.5 * (bst.predict(X_te_orig, raw_score=True) - bst.predict(X_te_mirr, raw_score=True)) / 5.0
        oof_z += c_oof / n_cfg
        if has_te:
            te_z += c_te / n_cfg
    return oof_z, te_z, imp, booster_strs


class AntisymResMLP(nn.Module):
    """Exact Blue/Red-antisymmetric residual MLP: lin(anti) + 0.5*(f(anti, sym) - f(-anti, sym)) + side_bias."""
    def __init__(self, d_anti: int, d_sym: int, hidden: int = 80, dropout: float = 0.38):
        super().__init__()
        self.lin = nn.Linear(d_anti, 1, bias=False)
        self.net = nn.Sequential(
            nn.Linear(d_anti + d_sym, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.side_bias = nn.Parameter(torch.zeros(1))

    def forward(self, anti, sym):
        h_pos = self.net(torch.cat([anti, sym], dim=-1))
        h_neg = self.net(torch.cat([-anti, sym], dim=-1))
        return (self.lin(anti) + 0.5 * (h_pos - h_neg)).squeeze(-1) + self.side_bias

    def export_numpy(self, a_sd, s_mu, s_sd, temp: float) -> dict:
        st = self.state_dict()
        return {
            "a_sd": a_sd.astype(np.float64),
            "s_mu": s_mu.astype(np.float64),
            "s_sd": s_sd.astype(np.float64),
            "temp": float(temp),
            "lin_w": st["lin.weight"].cpu().numpy().astype(np.float32),
            "w0": st["net.0.weight"].cpu().numpy().astype(np.float32),
            "b0": st["net.0.bias"].cpu().numpy().astype(np.float32),
            "ln1_w": st["net.1.weight"].cpu().numpy().astype(np.float32),
            "ln1_b": st["net.1.bias"].cpu().numpy().astype(np.float32),
            "w4": st["net.4.weight"].cpu().numpy().astype(np.float32),
            "b4": st["net.4.bias"].cpu().numpy().astype(np.float32),
            "ln5_w": st["net.5.weight"].cpu().numpy().astype(np.float32),
            "ln5_b": st["net.5.bias"].cpu().numpy().astype(np.float32),
            "w8": st["net.8.weight"].cpu().numpy().astype(np.float32),
            "b8": st["net.8.bias"].cpu().numpy().astype(np.float32),
            "side_bias": float(st["side_bias"].cpu().item()),
        }


def _train_antisym_mlp(anti_tr, sym_tr, y_tr, sw_tr, fold_ids, anti_te, sym_te, epochs: int = 18):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tr = len(y_tr)
    has_te = anti_te is not None
    oof_z = np.zeros(n_tr, dtype=np.float64)
    te_z = np.zeros(len(anti_te), dtype=np.float64) if has_te else None
    mlp_folds = []

    for f in range(5):
        torch.manual_seed(700 + f)
        np.random.seed(700 + f)
        vm = (fold_ids == f)
        tm = ~vm

        a_sd = np.maximum(np.std(anti_tr[tm], axis=0), 1e-5)
        s_mu = np.mean(sym_tr[tm], axis=0)
        s_sd = np.maximum(np.std(sym_tr[tm], axis=0), 1e-5)

        at = torch.from_numpy(np.clip(anti_tr[tm] / a_sd, -6.0, 6.0).astype(np.float32)).to(device)
        st = torch.from_numpy(np.clip((sym_tr[tm] - s_mu) / s_sd, -6.0, 6.0).astype(np.float32)).to(device)
        yt = torch.from_numpy(y_tr[tm].astype(np.float32)).to(device)
        wt = torch.from_numpy(sw_tr[tm].astype(np.float32)).to(device)

        av = torch.from_numpy(np.clip(anti_tr[vm] / a_sd, -6.0, 6.0).astype(np.float32)).to(device)
        sv = torch.from_numpy(np.clip((sym_tr[vm] - s_mu) / s_sd, -6.0, 6.0).astype(np.float32)).to(device)
        yv = torch.from_numpy(y_tr[vm].astype(np.float32)).to(device)
        wv = torch.from_numpy(sw_tr[vm].astype(np.float32)).to(device)

        model = AntisymResMLP(anti_tr.shape[1], sym_tr.shape[1], hidden=80, dropout=0.38).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.06)
        bs = 512
        n_b = len(at)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * math.ceil(n_b / bs), eta_min=5e-5)

        best_ll, best_st = 1e9, None
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_b, device=device)
            for i in range(0, n_b, bs):
                idx = perm[i:i + bs]
                z = model(at[idx], st[idx])
                w_n = wt[idx] / wt[idx].mean().clamp_min(1e-6)
                loss = (F.binary_cross_entropy_with_logits(z, yt[idx], reduction="none") * w_n).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()

            model.eval()
            with torch.no_grad():
                zv = model(av, sv)
                vll = float((F.binary_cross_entropy_with_logits(zv, yv, reduction="none") * wv).sum().item() / wv.sum().item())
            if vll < best_ll:
                best_ll = vll
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            zv_np = model(av, sv).cpu().numpy().astype(np.float64)
        t_opt = _fit_temperature(zv_np, y_tr[vm], sw_tr[vm])
        oof_z[vm] = zv_np / t_opt
        mlp_folds.append(model.export_numpy(a_sd, s_mu, s_sd, t_opt))

        if has_te:
            ae = torch.from_numpy(np.clip(anti_te / a_sd, -6.0, 6.0).astype(np.float32)).to(device)
            se = torch.from_numpy(np.clip((sym_te - s_mu) / s_sd, -6.0, 6.0).astype(np.float32)).to(device)
            with torch.no_grad():
                te_z += (model(ae, se).cpu().numpy().astype(np.float64) / t_opt) / 5.0

    return oof_z, te_z, mlp_folds


def run_stacker(db_path="data/patch_matrix.db", test_patch: str = "none", out_bundle="data/stacker_v2_bundle.pkl"):
    if not os.path.exists("data/v4_oof.npz") or not os.path.exists("data/draft_transformer_ensemble.pt"):
        raise RuntimeError("Run `python -m ml.train_transformer none` (or `<patch>`) first.")

    prod_mode = (test_patch is None) or (test_patch.lower() == "none")

    oof = np.load("data/v4_oof.npz", allow_pickle=True)
    tr_mids = oof["match_ids"].tolist()
    tr_patches = oof["patch_tags"].tolist()
    fold_ids = oof["fold_ids"].astype(np.int64)
    y_tr = oof["y_win"].astype(np.float64)
    mid_to_idx = {m: i for i, m in enumerate(tr_mids)}

    with open("data/transformer_vocab.json") as f:
        cid_to_idx = {int(k): int(v) for k, v in json.load(f)["cid_to_idx"].items()}
    num_champs = max(250, len(cid_to_idx) + 5)
    num_rc = 5 * num_champs + 1

    con = sqlite3.connect(db_path, timeout=30.0)
    patch_counts = dict(con.execute("SELECT patch, COUNT(*) FROM matches GROUP BY patch").fetchall())
    excluded = set() if prod_mode else {p for p in patch_counts if _patch_key(p) >= _patch_key(test_patch)}
    train_patches_sorted = sorted({p for p in tr_patches}, key=_patch_key, reverse=True)
    patch_age = {p: idx for idx, p in enumerate(train_patches_sorted)}

    engine = load_engine_from_db(db_path, exclude_patches=excluded)

    n_tr = len(tr_mids)
    b_rc_tr = np.zeros((n_tr, 5), dtype=np.int64)
    r_rc_tr = np.zeros((n_tr, 5), dtype=np.int64)
    b_gd_tr = np.zeros((n_tr, 5), dtype=np.float64)
    r_gd_tr = np.zeros((n_tr, 5), dtype=np.float64)
    b_xpd_tr = np.zeros((n_tr, 5), dtype=np.float64)
    r_xpd_tr = np.zeros((n_tr, 5), dtype=np.float64)
    mv_tr = np.zeros((n_tr, 10, MECH_DIM), dtype=np.float32)
    elo_tr = np.zeros(n_tr, dtype=np.int64)
    dur_tr = np.full(n_tr, 30.0, dtype=np.float64)
    obj_tr = np.zeros((n_tr, 2), dtype=np.float64)

    sw_long = np.array([0.95 ** patch_age[p] for p in tr_patches], dtype=np.float64)
    sw_tr = np.array([0.86 ** patch_age[p] for p in tr_patches], dtype=np.float64)
    sw_fast = np.array([0.55 ** patch_age[p] for p in tr_patches], dtype=np.float64)

    q_tr = ",".join("?" * len(train_patches_sorted))
    for mid, elo, dur, ft, gb, gr in con.execute(
        f"SELECT match_id, elo_tier, duration_sec, first_tower_blue, grubs_blue, grubs_red FROM matches WHERE patch IN ({q_tr})",
        train_patches_sorted,
    ):
        i = mid_to_idx.get(mid)
        if i is not None:
            elo_tr[i] = ELO_TO_IDX.get(elo, 3)
            dur_tr[i] = float(dur or 1800) / 60.0
            obj_tr[i, 0] = float(np.clip(ft or 0, 0, 1))
            obj_tr[i, 1] = float(np.clip(((gb or 0) - (gr or 0)) / 6.0, -1.0, 1.0))

    for mid, tid, role, cid, gd, xpd in con.execute(f"""
        SELECT p.match_id, p.team_id, p.role, p.cid, p.gd15, p.xpd15
        FROM match_participants p JOIN matches m ON p.match_id = m.match_id
        WHERE m.patch IN ({q_tr})
    """, train_patches_sorted):
        i = mid_to_idx.get(mid)
        if i is None:
            continue
        ri = ROLE_TO_IDX.get(role)
        if ri is None:
            continue
        cidx = cid_to_idx.get(cid, 0)
        rc = ri * num_champs + cidx if cidx > 0 else 0
        prof = engine.champions.get((role, cid))
        if tid == 100:
            b_rc_tr[i, ri] = rc; b_gd_tr[i, ri] = float(gd or 0.0); b_xpd_tr[i, ri] = float(xpd or 0.0)
            if prof is not None:
                mv_tr[i, ri] = prof.mech_vec.astype(np.float32)
        else:
            r_rc_tr[i, ri] = rc; r_gd_tr[i, ri] = float(gd or 0.0); r_xpd_tr[i, ri] = float(xpd or 0.0)
            if prof is not None:
                mv_tr[i, 5 + ri] = prof.mech_vec.astype(np.float32)

    # Optional Zero-Touch Holdout Patch
    c_te = b_rc_te = r_rc_te = mv_te = elo_te = y_te = None
    if not prod_mode:
        te_matches = con.execute("SELECT match_id, elo_tier, blue_win FROM matches WHERE patch = ?", (test_patch,)).fetchall()
        te_parts = defaultdict(list)
        for mid, tid, role, cid in con.execute("""
            SELECT p.match_id, p.team_id, p.role, p.cid
            FROM match_participants p JOIN matches m ON p.match_id = m.match_id
            WHERE m.patch = ?
        """, (test_patch,)):
            te_parts[mid].append((tid, role, cid))
        te_c_l, te_brc_l, te_rrc_l, te_mv_l, te_elo_l, te_y_l = [], [], [], [], [], []
        for mid, elo, bwin in te_matches:
            plist = te_parts.get(mid)
            if not plist or len(plist) != 10:
                continue
            c10 = np.zeros(10, dtype=np.int64)
            brc, rrc = np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64)
            mv10 = np.zeros((10, MECH_DIM), dtype=np.float32)
            bc = rc_cnt = 0
            for tid, role, cid in plist:
                ri = ROLE_TO_IDX.get(role)
                if ri is None:
                    continue
                cidx = cid_to_idx.get(cid, 0)
                rc = ri * num_champs + cidx if cidx > 0 else 0
                prof = engine.champions.get((role, cid))
                if tid == 100:
                    bc += 1; c10[ri] = cidx; brc[ri] = rc
                    if prof is not None:
                        mv10[ri] = prof.mech_vec.astype(np.float32)
                else:
                    rc_cnt += 1; c10[5 + ri] = cidx; rrc[ri] = rc
                    if prof is not None:
                        mv10[5 + ri] = prof.mech_vec.astype(np.float32)
            if bc == 5 and rc_cnt == 5:
                te_c_l.append(c10); te_brc_l.append(brc); te_rrc_l.append(rrc)
                te_mv_l.append(mv10); te_elo_l.append(ELO_TO_IDX.get(elo, 3)); te_y_l.append(float(bwin))
        c_te = np.array(te_c_l, dtype=np.int64)
        b_rc_te = np.array(te_brc_l, dtype=np.int64)
        r_rc_te = np.array(te_rrc_l, dtype=np.int64)
        mv_te = np.array(te_mv_l, dtype=np.float32)
        elo_te = np.array(te_elo_l, dtype=np.int64)
        y_te = np.array(te_y_l, dtype=np.float64)
    con.close()

    mode_str = f"FULL PRODUCTION ({len(train_patches_sorted)} patches, n={n_tr:,})" if prod_mode else f"ZERO-TOUCH HOLDOUT {test_patch}"
    print(f"[Stacker v2] Mode: {mode_str}")

    if not prod_mode:
        te_w, te_dw, te_dc, te_tm, te_ob = _run_v4_on_test(c_te, mv_te, elo_te)

    print("[Stacker v2] Building 5-fold OOF Tabular + Cross/Synergy GD@15 + Soft-Impute SVD...")
    tab_anti_tr = tab_sym_tr = None
    tab_anti_names = tab_sym_names = None
    for f in range(5):
        vm = (fold_ids == f)
        tm = ~vm
        f_stats = _fit_fold_stats(
            b_rc_tr[tm], r_rc_tr[tm], y_tr[tm], b_gd_tr[tm], r_gd_tr[tm],
            b_xpd_tr[tm], r_xpd_tr[tm], dur_tr[tm], obj_tr[tm],
            sw_tr[tm], sw_long[tm], sw_fast[tm], num_rc,
        )
        a_f, s_f, an, sn, _ = extract_tabular_block(b_rc_tr[vm], r_rc_tr[vm], mv_tr[vm], f_stats)
        if tab_anti_tr is None:
            tab_anti_tr = np.zeros((n_tr, a_f.shape[1]), dtype=np.float64)
            tab_sym_tr = np.zeros((n_tr, s_f.shape[1]), dtype=np.float64)
            tab_anti_names, tab_sym_names = an, sn
        tab_anti_tr[vm] = a_f
        tab_sym_tr[vm] = s_f

    print("[Stacker v2] Fitting 100%-data production lookup tables & SVD matrices...")
    full_stats = _fit_fold_stats(
        b_rc_tr, r_rc_tr, y_tr, b_gd_tr, r_gd_tr, b_xpd_tr, r_xpd_tr, dur_tr, obj_tr,
        sw_tr, sw_long, sw_fast, num_rc,
    )
    tab_anti_te = tab_sym_te = None
    if not prod_mode:
        tab_anti_te, tab_sym_te, _, _, _ = extract_tabular_block(b_rc_te, r_rc_te, mv_te, full_stats)

    nn_anti_tr, nn_sym_tr, nn_an, nn_sn, centers = extract_neural_block(
        oof["oof_win"], oof["oof_dur_win"], oof["oof_dur_cls"], oof["oof_tempo"], oof["oof_obj"]
    )
    nn_anti_te = nn_sym_te = None
    if not prod_mode:
        nn_anti_te, nn_sym_te, _, _, _ = extract_neural_block(te_w, te_dw, te_dc, te_tm, te_ob, centers=centers)

    elo_tr_col = elo_tr.reshape(-1, 1).astype(np.float64)
    elo_te_col = elo_te.reshape(-1, 1).astype(np.float64) if not prod_mode else None

    lgb_configs = [
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.015,
            "num_leaves": 7, "max_depth": 3, "min_child_samples": 350, "subsample": 0.80,
            "subsample_freq": 1, "colsample_bytree": 0.55, "reg_alpha": 10.0, "reg_lambda": 60.0,
            "min_gain_to_split": 0.02, "n_jobs": -1, "verbose": -1,
        },
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.015,
            "num_leaves": 18, "max_depth": 5, "min_child_samples": 280, "subsample": 0.78,
            "subsample_freq": 1, "colsample_bytree": 0.45, "reg_alpha": 12.0, "reg_lambda": 55.0,
            "min_gain_to_split": 0.03, "n_jobs": -1, "verbose": -1,
        },
    ]

    print("[Stacker v2] Training Family 2: Multi-Depth Pure-Tabular Antisymmetric LightGBM...")
    tab_names = tab_anti_names + tab_sym_names + ["elo_idx", "side_flag"]
    tab_oof_z, tab_te_z, _, tab_lgb_strs = _train_antisym_lgb(
        tab_anti_tr, np.hstack([tab_sym_tr, elo_tr_col]), y_tr, sw_tr, fold_ids,
        tab_anti_te, np.hstack([tab_sym_te, elo_te_col]) if not prod_mode else None,
        tab_names, lgb_configs,
    )

    print("[Stacker v2] Training Family 3: Multi-Depth Stacked Antisymmetric LightGBM...")
    full_anti_tr = np.hstack([nn_anti_tr, tab_anti_tr])
    full_sym_tr = np.hstack([nn_sym_tr, tab_sym_tr, elo_tr_col])
    full_anti_te = np.hstack([nn_anti_te, tab_anti_te]) if not prod_mode else None
    full_sym_te = np.hstack([nn_sym_te, tab_sym_te, elo_te_col]) if not prod_mode else None
    full_names = nn_an + tab_anti_names + nn_sn + tab_sym_names + ["elo_idx", "side_flag"]

    stk_oof_z, stk_te_z, stk_imp, stk_lgb_strs = _train_antisym_lgb(
        full_anti_tr, full_sym_tr, y_tr, sw_tr, fold_ids,
        full_anti_te, full_sym_te, full_names, lgb_configs,
    )

    print("[Stacker v2] Training Family 4: Temperature-Calibrated Antisymmetric Residual MLP...")
    mlp_oof_z, mlp_te_z, mlp_folds = _train_antisym_mlp(
        full_anti_tr, full_sym_tr, y_tr, sw_tr, fold_ids, full_anti_te, full_sym_te, epochs=18
    )

    # Clean 4-Family Meta-Blender (v4_NN, Tab_LGB, Stack_LGB, Calibrated_ResMLP)
    X_meta_tr = np.column_stack([oof["oof_win"], tab_oof_z, stk_oof_z, mlp_oof_z])
    meta_oof_p = np.zeros(n_tr, dtype=np.float64)
    for f in range(5):
        vm = (fold_ids == f)
        wm = _fit_dense_logreg(X_meta_tr[~vm], y_tr[~vm], l2=30.0)
        meta_oof_p[vm] = _predict_dense_logreg(X_meta_tr[vm], wm)
    w_meta = _fit_dense_logreg(X_meta_tr, y_tr, l2=30.0)

    print(f"Meta-Blend Weights [bias, v4_NN, Tab_LGB, Stack_LGB, Cal_ResMLP]: {np.round(w_meta, 3).tolist()}")
    print(f"ResMLP Fold Temperatures: {[round(m['temp'], 2) for m in mlp_folds]}")

    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
    p_v4_oof, p_tab_oof, p_stk_oof, p_mlp_oof = map(sig, (oof["oof_win"], tab_oof_z, stk_oof_z, mlp_oof_z))

    _print_table(
        f"ALL-TRAIN 5-FOLD LEAK-FREE OOF (n={n_tr:,}, {len(train_patches_sorted)} Patches)",
        [
            ("1. v4 Standalone 5x Transformer (OOF)", _metrics(y_tr, p_v4_oof)),
            ("2. Pure Tabular Antisym LGBM v2 (GD15+SVD)", _metrics(y_tr, p_tab_oof)),
            ("3. Stacked Antisym LGBM v2 (NN+Full Tab)", _metrics(y_tr, p_stk_oof)),
            ("4. Calibrated Antisym ResMLP (NN+Full Tab)", _metrics(y_tr, p_mlp_oof)),
            ("5. 4-Family Production Meta-Ensemble", _metrics(y_tr, meta_oof_p)),
        ],
    )
    _print_sig("4-Family Meta vs v4 Neural (All)", y_tr, p_v4_oof, meta_oof_p)

    for show_p in train_patches_sorted[:2]:
        m_p = np.array([p == show_p for p in tr_patches])
        if m_p.sum() >= 2000:
            _print_table(
                f"SAME-PATCH OOF BENCHMARK: Patch {show_p} (n={int(m_p.sum()):,})",
                [
                    ("1. v4 Standalone 5x Transformer (OOF)", _metrics(y_tr[m_p], p_v4_oof[m_p])),
                    ("2. Stacked Antisym LGBM v2", _metrics(y_tr[m_p], p_stk_oof[m_p])),
                    ("3. Calibrated Antisym ResMLP", _metrics(y_tr[m_p], p_mlp_oof[m_p])),
                    ("4. 4-Family Production Meta-Ensemble", _metrics(y_tr[m_p], meta_oof_p[m_p])),
                ],
            )
            _print_sig(f"4-Family Meta vs v4 NN ({show_p})", y_tr[m_p], p_v4_oof[m_p], meta_oof_p[m_p])

    if not prod_mode:
        X_meta_te = np.column_stack([te_w, tab_te_z, stk_te_z, mlp_te_z])
        meta_te_p = _predict_dense_logreg(X_meta_te, w_meta)
        p_naive_te = np.full_like(y_te, float(y_tr.mean()))
        p_v4_te, p_tab_te, p_stk_te, p_mlp_te = map(sig, (te_w, tab_te_z, stk_te_z, mlp_te_z))
        _print_table(
            f"100% ZERO-TOUCH FUTURE-PATCH HOLDOUT: Patch {test_patch} (n={len(y_te):,})",
            [
                ("Ref: Naive Side Prior", _metrics(y_te, p_naive_te)),
                ("1. v4 Standalone 5x Transformer", _metrics(y_te, p_v4_te)),
                ("2. Pure Tabular Antisym LGBM v2", _metrics(y_te, p_tab_te)),
                ("3. Stacked Antisym LGBM v2", _metrics(y_te, p_stk_te)),
                ("4. Calibrated Antisym ResMLP", _metrics(y_te, p_mlp_te)),
                ("5. 4-Family Production Meta-Ensemble", _metrics(y_te, meta_te_p)),
            ],
        )
        _print_sig("4-Family Meta vs Naive (Zero-Touch)", y_te, p_naive_te, meta_te_p)
        _print_sig("4-Family Meta vs v4 NN (Zero-Touch)", y_te, p_v4_te, meta_te_p)

    # Export portable live inference bundle (zero PyTorch required at runtime)
    os.makedirs(os.path.dirname(out_bundle), exist_ok=True)
    bundle = {
        "num_champs": num_champs,
        "train_patches": train_patches_sorted,
        "full_stats": full_stats,
        "nn_centers": centers,
        "w_meta": w_meta,
        "tab_lgb_strs": tab_lgb_strs,
        "stk_lgb_strs": stk_lgb_strs,
        "mlp_folds": mlp_folds,
    }
    with open(out_bundle, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(out_bundle) / (1024 * 1024)
    print(f"\nExported Live Stacker v2 Bundle -> {out_bundle} ({mb:.1f} MB, patches: {', '.join(train_patches_sorted)}).")


if __name__ == "__main__":
    tp = sys.argv[1] if len(sys.argv) > 1 else "none"
    run_stacker(test_patch=tp)