import os, pickle
import numpy as np


def build_wide_pair_indices(
    b_rc: np.ndarray,
    r_rc: np.ndarray,
    num_rc: int,
    pair_vocab: dict = None,
    pair_vk: np.ndarray = None,
    pair_vv: np.ndarray = None,
    min_cnt: int = 25,
):
    """Encodes 10 champs + 15 high-signal pairs into signed index arrays (partial-draft safe)."""
    n = len(b_rc)
    enemy_specs = [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1), (3, 4, 2), (4, 3, 2)]
    ally_specs = [(3, 4, 3), (1, 2, 3), (0, 1, 3), (1, 3, 3)]

    raw_keys, raw_sgns, raw_valid = [], [], []
    M = num_rc + 1
    for r1, r2, tag in enemy_specs:
        a, b = b_rc[:, r1], r_rc[:, r2]
        v = (a > 0) & (b > 0)
        swap = a > b
        lo = np.where(swap, b, a)
        hi = np.where(swap, a, b)
        sgn = np.where(swap, -1.0, 1.0)
        raw_keys.append(tag * (M * M) + lo * M + hi)
        raw_sgns.append(np.where(v, sgn, 0.0))
        raw_valid.append(v)

    for r1, r2, tag in ally_specs:
        for team_rc, t_sgn in ((b_rc, 1.0), (r_rc, -1.0)):
            a, b = team_rc[:, r1], team_rc[:, r2]
            v = (a > 0) & (b > 0)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            raw_keys.append(tag * (M * M) + lo * M + hi)
            raw_sgns.append(np.where(v, t_sgn, 0.0))
            raw_valid.append(v)

    pair_key_mat = np.column_stack(raw_keys)
    pair_sgn_mat = np.column_stack(raw_sgns)
    pair_val_mat = np.column_stack(raw_valid)

    if pair_vk is None and pair_vocab is None:
        valid_keys = pair_key_mat[pair_val_mat]
        uk, cnts = np.unique(valid_keys, return_counts=True)
        keep = uk[cnts >= min_cnt]
        pair_vocab = {int(k): num_rc + i for i, k in enumerate(keep)}

    if pair_vk is None:
        if pair_vocab:
            pair_vk = np.array(sorted(pair_vocab.keys()), dtype=np.int64)
            pair_vv = np.array([pair_vocab[int(k)] for k in pair_vk], dtype=np.int64)
        else:
            pair_vk = np.empty(0, dtype=np.int64)
            pair_vv = np.empty(0, dtype=np.int64)

    if len(pair_vk) > 0:
        pos = np.searchsorted(pair_vk, pair_key_mat)
        pos_clip = np.minimum(pos, len(pair_vk) - 1)
        matched = pair_val_mat & (pair_vk[pos_clip] == pair_key_mat)
        p_idx = np.where(matched, pair_vv[pos_clip], 0)
        p_sgn = np.where(matched, pair_sgn_mat, 0.0)
    else:
        p_idx = np.zeros_like(pair_key_mat)
        p_sgn = np.zeros_like(pair_sgn_mat)

    idx_mat = np.hstack([b_rc, r_rc, p_idx]).astype(np.int64)
    sgn_mat = np.hstack([np.ones((n, 5)), -np.ones((n, 5)), p_sgn]).astype(np.float64)
    D = num_rc + len(pair_vk)
    return idx_mat, sgn_mat, pair_vocab, D


def extract_tabular_block(b_rc: np.ndarray, r_rc: np.ndarray, mv_arr: np.ndarray, stats: dict):
    """Extracts purely tabular antisymmetric & symmetric features (100% partial-draft safe)."""
    n = len(b_rc)
    num_rc = stats["num_rc"]

    jbt_long = stats["w_jbt_long"][b_rc].sum(axis=1) - stats["w_jbt_long"][r_rc].sum(axis=1)
    jbt_med = stats["w_jbt_med"][b_rc].sum(axis=1) - stats["w_jbt_med"][r_rc].sum(axis=1)
    jbt_fast = stats["w_jbt_fast"][b_rc].sum(axis=1) - stats["w_jbt_fast"][r_rc].sum(axis=1)
    jbt_shift_med = jbt_fast - jbt_med
    jbt_shift_long = jbt_fast - jbt_long

    jridge_gd = stats["w_gd"][b_rc].sum(axis=1) - stats["w_gd"][r_rc].sum(axis=1)
    jridge_xpd = stats["w_xpd"][b_rc].sum(axis=1) - stats["w_xpd"][r_rc].sum(axis=1)
    jridge_ft = stats["w_ft"][b_rc].sum(axis=1) - stats["w_ft"][r_rc].sum(axis=1)
    jridge_grb = stats["w_grb"][b_rc].sum(axis=1) - stats["w_grb"][r_rc].sum(axis=1)

    idx_w, sgn_w, _, _ = build_wide_pair_indices(
        b_rc, r_rc, num_rc,
        pair_vocab=stats.get("pair_vocab"),
        pair_vk=stats.get("pair_vk"),
        pair_vv=stats.get("pair_vv"),
    )
    tw_win = stats["theta_wide_win"]
    tw_gd = stats["theta_wide_gd"]
    wide_pair_win = (tw_win[:-1][idx_w] * sgn_w).sum(axis=1)
    wide_pair_gd = (tw_gd[:-1][idx_w] * sgn_w).sum(axis=1)
    wide_pair_win_full = wide_pair_win + tw_win[-1]
    wide_pair_delta = wide_pair_win - jbt_med

    base_gd_diff = stats["base_gd"][b_rc].sum(axis=1) - stats["base_gd"][r_rc].sum(axis=1)
    base_xpd_diff = stats["base_xpd"][b_rc].sum(axis=1) - stats["base_xpd"][r_rc].sum(axis=1)
    scale_slope_diff = stats["scale_slope"][b_rc].sum(axis=1) - stats["scale_slope"][r_rc].sum(axis=1)
    scale_slope_sum = stats["scale_slope"][b_rc].sum(axis=1) + stats["scale_slope"][r_rc].sum(axis=1)

    # Direct-lane EB + Soft-Impute SVD
    lgd = np.zeros((n, 5), dtype=np.float64)
    lxpd = np.zeros((n, 5), dtype=np.float64)
    lwin = np.zeros((n, 5), dtype=np.float64)
    lobj = np.zeros((n, 5), dtype=np.float64)
    svd_lgd = np.zeros((n, 5), dtype=np.float64)
    svd_lxpd = np.zeros((n, 5), dtype=np.float64)
    svd_lwin = np.zeros((n, 5), dtype=np.float64)

    gmap, xmap, wmap, omap = stats["lane_gd_map"], stats["lane_xpd_map"], stats["lane_win_map"], stats["lane_obj_map"]
    for ri in range(5):
        keys = (b_rc[:, ri] * num_rc + r_rc[:, ri]).tolist()
        lgd[:, ri] = [gmap.get(k, 0.0) for k in keys]
        lxpd[:, ri] = [xmap.get(k, 0.0) for k in keys]
        lwin[:, ri] = [wmap.get(k, 0.0) for k in keys]
        lobj[:, ri] = [omap.get(k, 0.0) for k in keys]

        rmap = stats["role_comp_maps"][ri]
        i1, i2 = rmap[b_rc[:, ri]], rmap[r_rc[:, ri]]
        ok = (i1 >= 0) & (i2 >= 0)
        if ok.any():
            svd_lgd[ok, ri] = stats["role_svd_gd"][ri][i1[ok], i2[ok]]
            svd_lxpd[ok, ri] = stats["role_svd_xpd"][ri][i1[ok], i2[ok]]
            svd_lwin[ok, ri] = stats["role_svd_win"][ri][i1[ok], i2[ok]]

    # Cross-role Enemy EB + Soft-Impute SVD
    cross_win = np.zeros(n, dtype=np.float64)
    cross_gd = np.zeros(n, dtype=np.float64)
    cross_xpd = np.zeros(n, dtype=np.float64)
    cross_lgd = np.zeros(n, dtype=np.float64)
    svd_c_gd = np.zeros(n, dtype=np.float64)
    svd_c_wn = np.zeros(n, dtype=np.float64)
    bot2v2_cross_gd = np.zeros(n, dtype=np.float64)

    cw_m, cg_m, cx_m, cl_m = stats["cross_win_map"], stats["cross_gd_map"], stats["cross_xpd_map"], stats["cross_lgd_map"]
    hm_c, hm_s = stats["hm_c"], stats["hm_s"]
    rc_to_comp = stats["rc_to_comp"]
    b_comp = rc_to_comp[b_rc]
    r_comp = rc_to_comp[r_rc]

    for r1 in range(5):
        for r2 in range(5):
            if r1 != r2 and hm_c[r1, r2] >= 0.20:
                w_hm = hm_c[r1, r2]
                kb = (b_rc[:, r1] * num_rc + r_rc[:, r2]).tolist()
                kr = (r_rc[:, r1] * num_rc + b_rc[:, r2]).tolist()
                v_w = np.fromiter((cw_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((cw_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                v_g = np.fromiter((cg_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((cg_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                v_x = np.fromiter((cx_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((cx_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                v_l = np.fromiter((cl_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((cl_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                cross_win += 0.5 * w_hm * v_w
                cross_gd += 0.5 * w_hm * v_g
                cross_xpd += 0.5 * w_hm * v_x
                cross_lgd += 0.5 * w_hm * v_l
                if (r1, r2) in ((3, 4), (4, 3)):
                    bot2v2_cross_gd += 0.5 * v_l

                i1, i2 = b_comp[:, r1], r_comp[:, r2]
                ok = (i1 >= 0) & (i2 >= 0)
                if ok.any():
                    svd_c_gd[ok] += w_hm * stats["svd_cross_gd"][i1[ok], i2[ok]]
                    svd_c_wn[ok] += w_hm * stats["svd_cross_wn"][i1[ok], i2[ok]]

    # Ally Synergy EB + Soft-Impute SVD
    syn_win = np.zeros(n, dtype=np.float64)
    syn_gd = np.zeros(n, dtype=np.float64)
    syn_xpd = np.zeros(n, dtype=np.float64)
    syn_lgd = np.zeros(n, dtype=np.float64)
    syn_obj = np.zeros(n, dtype=np.float64)
    svd_s_gd = np.zeros(n, dtype=np.float64)
    svd_s_wn = np.zeros(n, dtype=np.float64)
    bot_syn_gd = np.zeros(n, dtype=np.float64)
    jgl_mid_syn_gd = np.zeros(n, dtype=np.float64)

    sw_m, sg_m, sx_m, sl_m, so_m = stats["syn_win_map"], stats["syn_gd_map"], stats["syn_xpd_map"], stats["syn_lgd_map"], stats["syn_obj_map"]
    for r1 in range(5):
        for r2 in range(r1 + 1, 5):
            if hm_s[r1, r2] >= 0.20:
                w_hm = hm_s[r1, r2]
                kb = (b_rc[:, r1] * num_rc + b_rc[:, r2]).tolist()
                kr = (r_rc[:, r1] * num_rc + r_rc[:, r2]).tolist()
                vw = np.fromiter((sw_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((sw_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                vg = np.fromiter((sg_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((sg_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                vx = np.fromiter((sx_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((sx_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                vl = np.fromiter((sl_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((sl_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                vo = np.fromiter((so_m.get(k, 0.0) for k in kb), dtype=np.float64, count=n) - np.fromiter((so_m.get(k, 0.0) for k in kr), dtype=np.float64, count=n)
                syn_win += w_hm * vw
                syn_gd += w_hm * vg
                syn_xpd += w_hm * vx
                syn_lgd += w_hm * vl
                syn_obj += w_hm * vo
                if (r1, r2) == (3, 4):
                    bot_syn_gd += vl
                elif (r1, r2) == (1, 2):
                    jgl_mid_syn_gd += vl

                bi1, bi2 = b_comp[:, r1], b_comp[:, r2]
                ri1, ri2 = r_comp[:, r1], r_comp[:, r2]
                okb = (bi1 >= 0) & (bi2 >= 0)
                okr = (ri1 >= 0) & (ri2 >= 0)
                if okb.any():
                    svd_s_gd[okb] += w_hm * stats["svd_syn_gd"][bi1[okb], bi2[okb]]
                    svd_s_wn[okb] += w_hm * stats["svd_syn_wn"][bi1[okb], bi2[okb]]
                if okr.any():
                    svd_s_gd[okr] -= w_hm * stats["svd_syn_gd"][ri1[okr], ri2[okr]]
                    svd_s_wn[okr] -= w_hm * stats["svd_syn_wn"][ri1[okr], ri2[okr]]

    # Comp Cliffs (damped cleanly on incomplete drafts; exact 1.0 on full 5v5)
    b_cnt = (b_rc > 0).sum(axis=1).astype(np.float64)
    r_cnt = (r_rc > 0).sum(axis=1).astype(np.float64)
    comp_scale = np.minimum(b_cnt, r_cnt) / 5.0

    bm, rm = mv_arr[:, :5, :].astype(np.float64), mv_arr[:, 5:, :].astype(np.float64)
    b_sum, r_sum = bm.sum(axis=1), rm.sum(axis=1)
    m_diff, m_tot = b_sum - r_sum, b_sum + r_sum
    b_tot_d = np.maximum(1e-5, b_sum[:, 1] + b_sum[:, 2] + b_sum[:, 3])
    r_tot_d = np.maximum(1e-5, r_sum[:, 1] + r_sum[:, 2] + r_sum[:, 3])
    b_phys_s, b_mag_s = b_sum[:, 1] / b_tot_d, b_sum[:, 2] / b_tot_d
    r_phys_s, r_mag_s = r_sum[:, 1] / r_tot_d, r_sum[:, 2] / r_tot_d

    mono_ad_cliff = comp_scale * (np.maximum(0.0, b_phys_s - 0.62) * (0.4 + r_sum[:, 4]) - np.maximum(0.0, r_phys_s - 0.62) * (0.4 + b_sum[:, 4]))
    mono_ap_cliff = comp_scale * (np.maximum(0.0, b_mag_s - 0.62) * (0.4 + r_sum[:, 4]) - np.maximum(0.0, r_mag_s - 0.62) * (0.4 + b_sum[:, 4]))
    cc_vs_mob = (b_sum[:, 5] * r_sum[:, 7]) - (r_sum[:, 5] * b_sum[:, 7])
    no_front_cliff = comp_scale * (np.maximum(0.0, 1.4 - b_sum[:, 4]) * r_sum[:, 0] - np.maximum(0.0, 1.4 - r_sum[:, 4]) * b_sum[:, 0])
    range_lane_diff = np.where((b_rc > 0) & (r_rc > 0), bm[:, :, 0] - rm[:, :, 0], 0.0)

    lgd_sum, lxpd_sum, lwin_sum, lobj_sum = lgd.sum(axis=1), lxpd.sum(axis=1), lwin.sum(axis=1), lobj.sum(axis=1)
    svd_lgd_sum, svd_lxpd_sum, svd_lwin_sum = svd_lgd.sum(axis=1), svd_lxpd.sum(axis=1), svd_lwin.sum(axis=1)
    lgd_skew = lgd.max(axis=1) + lgd.min(axis=1)
    lgd_spread = lgd.max(axis=1) - lgd.min(axis=1)
    early_x_scale = (jridge_gd + lgd_sum) * scale_slope_diff

    anti = np.column_stack([
        jbt_long, jbt_med, jbt_fast, jbt_shift_med, jbt_shift_long,
        wide_pair_win, wide_pair_gd, wide_pair_delta,
        jridge_gd, jridge_xpd, jridge_ft, jridge_grb,
        base_gd_diff, base_xpd_diff, scale_slope_diff,
        lgd, lxpd, lgd_sum, lxpd_sum, lwin_sum, lobj_sum, lgd_skew,
        svd_lgd_sum, svd_lxpd_sum, svd_lwin_sum,
        cross_win, cross_gd, cross_xpd, cross_lgd, bot2v2_cross_gd, svd_c_gd, svd_c_wn,
        syn_win, syn_gd, syn_xpd, syn_lgd, syn_obj, bot_syn_gd, jgl_mid_syn_gd, svd_s_gd, svd_s_wn,
        m_diff, mono_ad_cliff, mono_ap_cliff, cc_vs_mob, no_front_cliff, range_lane_diff,
    ])
    sym = np.column_stack([
        scale_slope_sum, lgd_spread, early_x_scale,
        m_tot[:, 0], m_tot[:, 4], m_tot[:, 5], m_tot[:, 6], m_tot[:, 9],
    ])
    anti_names = [
        "jbt_long", "jbt_med", "jbt_fast", "jbt_shift_med", "jbt_shift_long",
        "wide_pair_win", "wide_pair_gd15", "wide_pair_delta",
        "jridge_gd15", "jridge_xpd15", "jridge_ft", "jridge_grubs",
        "base_gd15_diff", "base_xpd15_diff", "scale_slope_diff",
        "eb_lgd_top", "eb_lgd_jgl", "eb_lgd_mid", "eb_lgd_bot", "eb_lgd_sup",
        "eb_lxpd_top", "eb_lxpd_jgl", "eb_lxpd_mid", "eb_lxpd_bot", "eb_lxpd_sup",
        "eb_lgd_sum", "eb_lxpd_sum", "eb_lwin_sum", "eb_lobj_sum", "eb_lgd_skew",
        "svd_lgd_sum", "svd_lxpd_sum", "svd_lwin_sum",
        "eb_cross_win", "eb_cross_gd", "eb_cross_xpd", "eb_cross_lgd", "eb_bot2v2_cgd", "svd_cross_gd", "svd_cross_wn",
        "eb_syn_win", "eb_syn_gd", "eb_syn_xpd", "eb_syn_lgd", "eb_syn_obj", "eb_bot_sgd", "eb_jm_sgd", "svd_syn_gd", "svd_syn_wn",
        "md_range", "md_phys", "md_mag", "md_true", "md_mit", "md_cc", "md_hs", "md_mob", "md_cs", "md_scale",
        "cliff_mono_ad", "cliff_mono_ap", "cliff_cc_vs_mob", "cliff_no_front",
        "rng_top", "rng_jgl", "rng_mid", "rng_bot", "rng_sup",
    ]
    sym_names = [
        "scale_slope_sum", "eb_lgd_spread", "early_x_scale",
        "tot_range", "tot_mit", "tot_cc", "tot_hs", "tot_scale",
    ]
    return anti, sym, anti_names, sym_names, wide_pair_win_full


def extract_neural_block(nn_win, nn_dur_win, nn_dur_cls, nn_tempo, nn_obj, centers=None):
    if centers is None:
        centers = {
            "win": float(np.mean(nn_win)),
            "dur_win": np.mean(nn_dur_win, axis=0),
            "tempo": np.mean(nn_tempo, axis=0),
            "obj": np.mean(nn_obj, axis=0),
        }
    w_c = nn_win - centers["win"]
    dw_c = nn_dur_win - centers["dur_win"]
    tm_c = nn_tempo - centers["tempo"]
    ob_c = nn_obj - centers["obj"]

    p_dur = np.clip(nn_dur_cls, 1e-4, 1.0)
    p_dur = p_dur / p_dur.sum(axis=1, keepdims=True)
    exp_dur = p_dur @ np.array([22.5, 27.5, 32.5, 38.0], dtype=np.float64)
    p_marg = np.clip((p_dur * (1.0 / (1.0 + np.exp(-np.clip(dw_c, -12.0, 12.0))))).sum(axis=1), 1e-5, 1.0 - 1e-5)
    z_marg = np.log(p_marg / (1.0 - p_marg))
    dur_slope = dw_c[:, 3] - dw_c[:, 0]

    gd = tm_c[:, :, 0]
    xpd = tm_c[:, :, 1]
    gd_sum = gd.sum(axis=1)
    xpd_sum = xpd.sum(axis=1)
    gd_skew = gd.max(axis=1) + gd.min(axis=1)
    gd_spread = gd.max(axis=1) - gd.min(axis=1)
    early_late_int = dur_slope * (exp_dur - 30.0) / 5.0

    anti = np.column_stack([w_c, z_marg, dw_c, dur_slope, early_late_int, gd, xpd, gd_sum, xpd_sum, gd_skew, ob_c])
    sym = np.column_stack([p_dur, exp_dur, gd_spread, np.std(gd, axis=1)])
    anti_names = [
        "nn_win", "nn_dur_marg", "nn_dw0", "nn_dw1", "nn_dw2", "nn_dw3", "nn_dur_slope", "nn_scale_x_dur",
        "nn_gd_top", "nn_gd_jgl", "nn_gd_mid", "nn_gd_bot", "nn_gd_sup",
        "nn_xpd_top", "nn_xpd_jgl", "nn_xpd_mid", "nn_xpd_bot", "nn_xpd_sup",
        "nn_gd_sum", "nn_xpd_sum", "nn_gd_skew",
        "nn_obj_ft", "nn_obj_grubs", "nn_obj_soul", "nn_obj_baron",
    ]
    sym_names = ["nn_dp0", "nn_dp1", "nn_dp2", "nn_dp3", "nn_exp_dur", "nn_gd_spread", "nn_gd_std"]
    return anti, sym, anti_names, sym_names, centers


def _layernorm_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * w + b


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def eval_antisym_mlp_numpy(anti: np.ndarray, sym: np.ndarray, mlp_folds: list[dict]) -> np.ndarray:
    """Pure-NumPy inference for the 5-fold calibrated Antisymmetric Residual MLP (<1ms, zero PyTorch)."""
    out = np.zeros(len(anti), dtype=np.float64)
    for f in mlp_folds:
        a_n = np.clip(anti / f["a_sd"], -6.0, 6.0).astype(np.float32)
        s_n = np.clip((sym - f["s_mu"]) / f["s_sd"], -6.0, 6.0).astype(np.float32)

        def _branch(x):
            h = _silu_np(_layernorm_np(x @ f["w0"].T + f["b0"], f["ln1_w"], f["ln1_b"]))
            h = _silu_np(_layernorm_np(h @ f["w4"].T + f["b4"], f["ln5_w"], f["ln5_b"]))
            return (h @ f["w8"].T + f["b8"]).reshape(-1)

        hp = _branch(np.hstack([a_n, s_n]))
        hn = _branch(np.hstack([-a_n, s_n]))
        z = (a_n @ f["lin_w"].T).reshape(-1) + 0.5 * (hp - hn) + f["side_bias"]
        out += (z.astype(np.float64) / f["temp"]) / len(mlp_folds)
    return out


class StackerV2Runtime:
    """CPU-only (<15ms/batch) heterogeneous Stacker v2 runtime: Tabular LGBM + Stacked LGBM + Calibrated ResMLP."""
    def __init__(self, bundle_path="data/stacker_v2_bundle.pkl"):
        self.active = False
        if not os.path.exists(bundle_path):
            return
        try:
            import lightgbm as lgb
            with open(bundle_path, "rb") as f:
                b = pickle.load(f)
            self.num_champs = int(b["num_champs"])
            self.stats = b["full_stats"]
            self.nn_centers = b["nn_centers"]
            self.w_meta = np.asarray(b["w_meta"], dtype=np.float64)
            self.mlp_folds = b["mlp_folds"]
            self.tab_boosters = [lgb.Booster(model_str=s) for s in b["tab_lgb_strs"]]
            self.stk_boosters = [lgb.Booster(model_str=s) for s in b["stk_lgb_strs"]]
            self.patches_trained = b.get("train_patches", [])
            self.active = True
        except Exception:
            self.active = False

    def score_batch(
        self,
        c_arr: np.ndarray,
        mv_arr: np.ndarray,
        elo_arr: np.ndarray,
        nn_win: np.ndarray,
        nn_dur_win: np.ndarray,
        nn_dur_probs: np.ndarray,
        nn_tempo: np.ndarray,
        nn_obj: np.ndarray,
    ):
        n = len(c_arr)
        roles = np.arange(5, dtype=np.int64)[None, :]
        b_c, r_c = c_arr[:, :5], c_arr[:, 5:]
        b_rc = np.where(b_c > 0, roles * self.num_champs + b_c, 0)
        r_rc = np.where(r_c > 0, roles * self.num_champs + r_c, 0)

        tab_anti, tab_sym, _, _, _ = extract_tabular_block(b_rc, r_rc, mv_arr, self.stats)
        nn_anti, nn_sym, _, _, _ = extract_neural_block(
            nn_win, nn_dur_win, nn_dur_probs, nn_tempo, nn_obj, centers=self.nn_centers
        )

        elo_col = elo_arr.reshape(-1, 1).astype(np.float64)
        ones = np.ones((n, 1), dtype=np.float64)

        # Family 2: Pure-Tabular Antisymmetric LightGBM
        t_sym = np.hstack([tab_sym, elo_col])
        X_to = np.hstack([tab_anti, t_sym, ones])
        X_tm = np.hstack([-tab_anti, t_sym, -ones])
        tab_z = np.zeros(n, dtype=np.float64)
        for bst in self.tab_boosters:
            tab_z += 0.5 * (bst.predict(X_to, raw_score=True) - bst.predict(X_tm, raw_score=True))
        tab_z /= max(1, len(self.tab_boosters))

        # Family 3: Stacked Antisymmetric LightGBM
        full_anti = np.hstack([nn_anti, tab_anti])
        full_sym = np.hstack([nn_sym, tab_sym, elo_col])
        X_so = np.hstack([full_anti, full_sym, ones])
        X_sm = np.hstack([-full_anti, full_sym, -ones])
        stk_z = np.zeros(n, dtype=np.float64)
        for bst in self.stk_boosters:
            stk_z += 0.5 * (bst.predict(X_so, raw_score=True) - bst.predict(X_sm, raw_score=True))
        stk_z /= max(1, len(self.stk_boosters))

        # Family 4: Calibrated Antisymmetric Residual MLP (pure NumPy)
        mlp_z = eval_antisym_mlp_numpy(full_anti, full_sym, self.mlp_folds)

        # Calibrated 4-Family Meta-Blend: [bias, v4_NN, Tab_LGB, Stack_LGB, ResMLP]
        X_meta = np.column_stack([nn_win.astype(np.float64), tab_z, stk_z, mlp_z])
        meta_logit = self.w_meta[0] + X_meta @ self.w_meta[1:]
        family_std = np.std(X_meta, axis=1)
        return meta_logit, family_std