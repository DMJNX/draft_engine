import hashlib, json, math, os, pickle, sqlite3
from collections import defaultdict
import numpy as np

from core.scoring_engine import (
    ROLES, MATCHUP_HEATMAP, SYNERGY_HEATMAP,
    ChampProfile, DraftScoringEngine, PersonalProfile, SetupOption, logit, sigmoid
)
from data.multi_patch_model import build_decayed_residuals
from ml.onnx_value_net import ONNXValueNet

_L1_CACHE_VERSION = 3
ALL_ELO_TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER")


def _patch_sort_key(p: str):
    return tuple(int(x) for x in p.split(".") if x.isdigit())


def _estimate_otp_bias(con: sqlite3.Connection, excluded: set = None) -> dict:
    """Regresses win ~ log10(mastery_points + 100) per (role, cid) on the 10% mastery subsample."""
    by_rc = defaultdict(list)
    try:
        if excluded:
            q_ex = ",".join("?" * len(excluded))
            rows_iter = con.execute(f"""
                SELECT s.role, s.cid, s.mastery_pts, s.win
                FROM mastery_subsample s
                JOIN seen_matches m ON s.match_id = m.match_id
                WHERE m.patch NOT IN ({q_ex})
            """, list(excluded))
        else:
            rows_iter = con.execute("SELECT role, cid, mastery_pts, win FROM mastery_subsample")
        for role, cid, pts, win in rows_iter:
            by_rc[(role, cid)].append((math.log10(max(0, pts) + 100.0), float(win)))
    except Exception:
        return {}

    baseline_log_m = 4.0
    out = {}
    for key, rows in by_rc.items():
        n = len(rows)
        if n < 25:
            continue
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        x_mean = float(xs.mean())
        var_x = float(np.var(xs))
        if var_x < 0.05:
            continue
        raw_slope = float(np.cov(xs, ys, ddof=0)[0, 1] / var_x) * 4.0
        shrink = n / (n + 120.0)
        slope = float(np.clip(shrink * raw_slope + (1.0 - shrink) * 0.03, 0.0, 0.14))
        otp_bias = float(np.clip(slope * (x_mean - baseline_log_m), -0.06, 0.14))
        out[key] = (otp_bias, slope, baseline_log_m)
    return out


def _fit_joint_bt_ratings(
    con: sqlite3.Connection,
    patches: list[str],
    rework_cutoffs: dict[int, int],
    valid_rc_keys: set,
    decay: float = 0.88,
    l2: float = 350.0,
) -> dict:
    """
    Fast exact Newton-CG Joint L2 Bradley-Terry solver across all 10 champions simultaneously.
    Uses 10-hot integer indexing (~0.3s on 110k matches, <10MB RAM) and respects rework cutoffs.
    Returns {(role, cid): joint_strength_logit} centered around 0 (Blue side absorbed in intercept).
    """
    if not patches or not valid_rc_keys:
        return {}

    patch_age = {p: idx for idx, p in enumerate(patches)}
    q_marks = ",".join("?" * len(patches))
    match_rows = con.execute(
        f"SELECT match_id, patch, blue_win FROM matches WHERE patch IN ({q_marks})", patches
    ).fetchall()
    if len(match_rows) < 500:
        return {}

    parts_map = defaultdict(list)
    for mid, tid, role, cid in con.execute(f"""
        SELECT p.match_id, p.team_id, p.role, p.cid
        FROM match_participants p
        JOIN matches m ON p.match_id = m.match_id
        WHERE m.patch IN ({q_marks})
    """, patches):
        parts_map[mid].append((tid, role, cid))

    rc_list = sorted(valid_rc_keys)
    rc_to_idx = {rc: i + 1 for i, rc in enumerate(rc_list)}  # index 0 = reserved dummy padding
    D = len(rc_list) + 1

    b_rows, r_rows, y_list, sw_list = [], [], [], []
    for mid, patch, bwin in match_rows:
        plist = parts_map.get(mid)
        if not plist or len(plist) != 10:
            continue
        age = patch_age[patch]
        b_arr, r_arr = [], []
        for tid, role, cid in plist:
            idx = rc_to_idx.get((role, cid), 0) if age <= rework_cutoffs.get(cid, 999) else 0
            if tid == 100:
                b_arr.append(idx)
            elif tid == 200:
                r_arr.append(idx)
        if len(b_arr) == 5 and len(r_arr) == 5:
            b_rows.append(b_arr)
            r_rows.append(r_arr)
            y_list.append(float(bwin))
            sw_list.append(decay ** age)

    if len(y_list) < 500:
        return {}

    b_idx = np.array(b_rows, dtype=np.int32)
    r_idx = np.array(r_rows, dtype=np.int32)
    b_flat = b_idx.ravel()
    r_flat = r_idx.ravel()
    y = np.array(y_list, dtype=np.float64)
    sw = np.array(sw_list, dtype=np.float64)

    theta = np.zeros(D + 1, dtype=np.float64)

    for _ in range(5):
        w = theta[:D]
        b = theta[D]
        z = np.clip(w[b_idx].sum(axis=1) - w[r_idx].sum(axis=1) + b, -15.0, 15.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = sw * (p - y)
        W = np.clip(p * (1.0 - p), 1e-6, 0.25) * sw

        err5 = np.repeat(err, 5)
        grad = np.empty(D + 1, dtype=np.float64)
        grad[:D] = (
            np.bincount(b_flat, weights=err5, minlength=D)
            - np.bincount(r_flat, weights=err5, minlength=D)
            + l2 * w
        )
        grad[0] = 0.0
        grad[D] = float(err.sum()) + 1e-4 * b

        def matvec(v):
            vw = v[:D]
            vb = v[D]
            xv = vw[b_idx].sum(axis=1) - vw[r_idx].sum(axis=1) + vb
            wxv = W * xv
            wxv5 = np.repeat(wxv, 5)
            out = np.empty(D + 1, dtype=np.float64)
            out[:D] = (
                np.bincount(b_flat, weights=wxv5, minlength=D)
                - np.bincount(r_flat, weights=wxv5, minlength=D)
                + l2 * vw
            )
            out[0] = vw[0]
            out[D] = float(wxv.sum()) + 1e-4 * vb
            return out

        step = np.zeros(D + 1, dtype=np.float64)
        r_vec = -grad
        p_dir = r_vec.copy()
        rs_old = float(r_vec @ r_vec)
        for _ in range(18):
            if rs_old < 1e-10:
                break
            Ap = matvec(p_dir)
            denom = float(p_dir @ Ap)
            if denom <= 1e-12:
                break
            alpha = rs_old / denom
            step += alpha * p_dir
            r_vec -= alpha * Ap
            rs_new = float(r_vec @ r_vec)
            if rs_new < 1e-10:
                break
            p_dir = r_vec + (rs_new / rs_old) * p_dir
            rs_old = rs_new

        theta += np.clip(step, -1.0, 1.0)
        theta[0] = 0.0

    return {rc: float(theta[idx]) for rc, idx in rc_to_idx.items()}


def _compute_l1_fingerprint(con: sqlite3.Connection, settings_path: str) -> str:
    h = hashlib.sha256()
    h.update(f"v{_L1_CACHE_VERSION}".encode())
    for row in con.execute("SELECT patch, match_count FROM patch_totals ORDER BY patch"):
        h.update(f"|{row[0]}:{row[1] // 500}".encode())
    n_m = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    try:
        n_ms = con.execute("SELECT COUNT(*) FROM mastery_subsample").fetchone()[0]
    except Exception:
        n_ms = 0
    h.update(f"|m:{n_m // 500}|ms:{n_ms // 500}".encode())
    with open(settings_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_engine_from_db(
    db_path="data/patch_matrix.db",
    settings_path="config/user_settings.json",
    personal: PersonalProfile = None,
    exclude_patches: set = None,
) -> DraftScoringEngine:
    onnx_net = ONNXValueNet()
    if not os.path.exists(db_path):
        return build_demo_engine(personal, onnx_net)

    con = sqlite3.connect(db_path, timeout=30.0)
    excluded = set(exclude_patches or ())
    cache_path = os.path.splitext(db_path)[0] + "_l1_cache.pkl"
    fp = None

    if not excluded:
        fp = _compute_l1_fingerprint(con, settings_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("fp") == fp:
                    con.close()
                    return DraftScoringEngine(
                        cached["champions"], cached["pre_ctr"], cached["pre_syn"],
                        cached["ctr_kish"], cached["syn_kish"], cached["ban_likes"],
                        cached["setup_table"], personal, onnx_net,
                    )
            except Exception:
                pass

    with open(settings_path) as f:
        cfg = json.load(f)

    patches = sorted(
        [r[0] for r in con.execute("SELECT patch FROM patch_totals WHERE match_count > 0") if r[0] not in excluded],
        key=_patch_sort_key, reverse=True,
    )[:12]
    if not patches:
        con.close()
        return build_demo_engine(personal, onnx_net)

    patch_age = {p: idx for idx, p in enumerate(patches)}
    champ_meta = {
        r[0]: (r[1], r[2], r[3], r[4], r[5])
        for r in con.execute("SELECT cid, key_name, display_name, difficulty, attack_range, mobility_prior FROM champ_meta")
    }
    key_to_id = {v[0]: k for k, v in champ_meta.items()}
    rune_names = dict(con.execute("SELECT rune_id, name FROM rune_meta"))
    patch_totals = dict(con.execute("SELECT patch, match_count FROM patch_totals"))
    otp_params = _estimate_otp_bias(con, excluded=excluded)

    rework_cutoffs = {
        key_to_id[k]: patch_age[p]
        for k, p in cfg.get("manual_rework_cutoffs", {}).items()
        if k in key_to_id and p in patch_age
    }

    q_marks = ",".join("?" * len(patches))
    champ_by_patch = defaultdict(dict)
    stat_accum = defaultdict(lambda: np.zeros(17))
    raw_wg = {}

    for row in con.execute(f"""
        SELECT patch, role, cid, wins, games,
               b0_wins, b0_games, b1_wins, b1_games, b2_wins, b2_games, b3_wins, b3_games,
               blue_wins, blue_games, phys_dpm, mag_dpm, true_dpm, mit_dpm, cc_pm, cs_pm, heal_shield_pm, gpm
        FROM champ_patch_stats WHERE patch IN ({q_marks})
    """, patches):
        p, role, cid, w, g, b0w, b0g, b1w, b1g, b2w, b2g, b3w, b3g, blw, blg, ph, mg, tr, mit, cc, cs, hs, gpm = row
        raw_wg[(p, role, cid)] = (w, g)
        stat_accum[(role, cid)] += np.array([g, ph, mg, tr, mit, cc, cs, hs, gpm, b0w, b0g, b1w, b1g, b2w, b2g, b3w, b3g])
        if g >= 5:
            champ_by_patch[(p, role)][cid] = (w + 100.0) / (g + 200.0)

    # Statistically significant rework tripwire (z >= 3.2 AND >= 4.5pp shift on >= 250 games/patch)
    all_rc_keys = {(r, c) for (_, r, c) in raw_wg}
    for role, cid in all_rc_keys:
        for t in range(len(patches) - 1):
            w1, g1 = raw_wg.get((patches[t], role, cid), (0, 0))
            w2, g2 = raw_wg.get((patches[t + 1], role, cid), (0, 0))
            if g1 >= 250 and g2 >= 250:
                p1, p2 = w1 / g1, w2 / g2
                diff = abs(p1 - p2)
                se = math.sqrt(0.25 / g1 + 0.25 / g2)
                if diff >= 0.045 and (diff / se) >= 3.2:
                    rework_cutoffs[cid] = min(rework_cutoffs.get(cid, 999), t)
                    break

    decay_base = 0.88
    k_base = 220.0
    latest_rows = {}
    for role, cid in all_rc_keys:
        max_age = rework_cutoffs.get(cid, 999)
        w_sum, g_sum, g_sq_sum, tot_m_sum = 0.0, 0.0, 0.0, 0.0
        for t, p in enumerate(patches):
            if t > max_age:
                break
            wt = decay_base ** t
            tot_m_sum += wt * patch_totals.get(p, 0)
            if (p, role, cid) in raw_wg:
                w, g = raw_wg[(p, role, cid)]
                w_sum += wt * w
                g_sum += wt * g
                g_sq_sum += (wt ** 2) * g
        if g_sum < 10.0 or tot_m_sum <= 0.0:
            continue
        n_kish = (g_sum ** 2) / max(g_sq_sum, 1e-6)
        raw_p = w_sum / g_sum
        shrunk_wr = (n_kish * raw_p + 0.5 * k_base) / (n_kish + k_base)
        pr = g_sum / max(1.0, 2.0 * tot_m_sum)
        latest_rows[(role, cid)] = (shrunk_wr, pr, int(round(n_kish)))

    eligible_rc = {k for k, (_, pr, _) in latest_rows.items() if pr >= 0.004}
    joint_logits = _fit_joint_bt_ratings(con, patches, rework_cutoffs, eligible_rc, decay=decay_base, l2=350.0)

    side_accum = defaultdict(lambda: [0, 0, 0, 0])
    for role, cid, w, g, bw, bg in con.execute(
        f"SELECT role, cid, wins, games, blue_wins, blue_games FROM champ_patch_stats WHERE patch IN ({q_marks})",
        patches,
    ):
        side_accum[(role, cid)][0] += bw; side_accum[(role, cid)][1] += bg
        side_accum[(role, cid)][2] += (w - bw); side_accum[(role, cid)][3] += (g - bg)

    elo_accum = defaultdict(lambda: [0, 0])
    for elo, role, cid, w, g in con.execute(
        f"SELECT elo_tier, role, cid, wins, games FROM elo_champ_stats WHERE patch IN ({q_marks})",
        patches,
    ):
        elo_accum[(elo, role, cid)][0] += w; elo_accum[(elo, role, cid)][1] += g

    elo_match_totals = dict(
        con.execute(
            f"SELECT elo_tier, COUNT(*) FROM matches WHERE patch IN ({q_marks}) GROUP BY elo_tier",
            patches,
        ).fetchall()
    )

    raw_comp, raw_mech = {}, {}
    for (role, cid), arr in stat_accum.items():
        g = max(1.0, arr[0])
        ph, mg, tr, mit, cc, cs, hs = arr[1]/g, arr[2]/g, arr[3]/g, arr[4]/g, arr[5]/g, arr[6]/g, arr[7]/g
        tot_d = max(1.0, ph + mg + tr)
        b_wrs = [(arr[9 + 2*i] + 15.0) / (arr[10 + 2*i] + 30.0) for i in range(4)]
        scale_sig = float(np.clip(0.5 + ((b_wrs[2] + b_wrs[3]) - (b_wrs[0] + b_wrs[1])) * 2.5, 0.0, 1.0))
        _, _, _, rng_val, mob_val = champ_meta.get(cid, ("", "", 5, 175.0, 0.4))

        raw_comp[(role, cid)] = np.array([ph, mg, mit, cc, cs, scale_sig])
        raw_mech[(role, cid)] = np.array([
            min(1.0, rng_val / 650.0), ph / tot_d, mg / tot_d, tr / tot_d,
            mit, cc, hs, mob_val, cs, scale_sig
        ])

    comp_p95 = np.maximum(np.percentile(np.array(list(raw_comp.values())), 95, axis=0), 1e-5) if raw_comp else np.ones(6)
    mech_p95 = np.maximum(np.percentile(np.array(list(raw_mech.values())), 95, axis=0), 1e-5) if raw_mech else np.ones(10)
    for idx in (0, 1, 2, 3, 7, 9):
        mech_p95[idx] = 1.0

    champions = {}
    marginal_wr_map = {}
    for (role, cid), (wr_marg, pr, n_kish_base) in latest_rows.items():
        if pr < 0.004 or (role, cid) not in raw_comp:
            continue
        marginal_wr_map[(role, cid)] = wr_marg
        wr_joint = sigmoid(joint_logits[(role, cid)]) if (role, cid) in joint_logits else wr_marg
        arr = stat_accum[(role, cid)]
        dur_wrs = np.array([(arr[9 + 2*i] + 15.0) / (arr[10 + 2*i] + 30.0) for i in range(4)])

        bw, bg, rw, rg = side_accum[(role, cid)]
        n_side = min(bg, rg)
        raw_b_delta = logit((bw + 50.0) / (bg + 100.0)) - logit((rw + 50.0) / (rg + 100.0))
        b_delta = (n_side / (n_side + 400.0)) * raw_b_delta

        elo_d, elo_pr = {}, {}
        for tier in ALL_ELO_TIERS:
            ew, eg = elo_accum[(tier, role, cid)]
            if eg >= 30:
                elo_wr = (ew + 80.0) / (eg + 160.0)
                elo_d[tier] = (eg / (eg + 180.0)) * (logit(elo_wr) - logit(wr_marg))
            t_tot = elo_match_totals.get(tier, 0)
            if t_tot >= 400:
                raw_pr = eg / max(1.0, 2.0 * t_tot)
                a_pr = t_tot / (t_tot + 1500.0)
                elo_pr[tier] = max(1e-4, a_pr * raw_pr + (1.0 - a_pr) * pr)

        _, name, diff_r, _, _ = champ_meta.get(cid, ("", f"Champ {cid}", 5, 175.0, 0.4))
        otp_b, m_slope, base_lm = otp_params.get((role, cid), (0.0, 0.03, 4.0))
        champions[(role, cid)] = ChampProfile(
            champ_id=cid, name=name, role=role, base_wr=wr_joint, base_games=int(n_kish_base), play_rate=pr,
            comp_vec=np.clip(raw_comp[(role, cid)] / comp_p95, 0.0, 1.0),
            mech_vec=np.clip(raw_mech[(role, cid)] / mech_p95, 0.0, 1.0),
            duration_wr=dur_wrs, blue_side_delta=b_delta, elo_wr_delta=elo_d, elo_play_rate=elo_pr,
            mechanical_floor=float(np.clip(diff_r / 10.0, 0.1, 1.0)),
            otp_bias_logit=otp_b, mastery_slope=m_slope, baseline_log_m=base_lm,
        )

    pre_ctr, pre_syn, ctr_kish, syn_kish = {}, {}, {}, {}
    for tbl, is_enemy, target, k_target, hm in (
        ("matchup_patch_stats", True, pre_ctr, ctr_kish, MATCHUP_HEATMAP),
        ("synergy_patch_stats", False, pre_syn, syn_kish, SYNERGY_HEATMAP),
    ):
        for r1 in ROLES:
            for r2 in ROLES:
                if not is_enemy and r1 == r2:
                    continue
                rows = con.execute(
                    f"SELECT patch, c1, c2, wins, games FROM {tbl} WHERE r1=? AND r2=? AND patch IN ({q_marks})",
                    [r1, r2, *patches],
                ).fetchall()
                if not rows:
                    continue
                by_p = {
                    p: {"base_wr_1": champ_by_patch[(p, r1)], "base_wr_2": champ_by_patch[(p, r2)], "pairs": {}}
                    for p in patches
                }
                for p, c1, c2, w, g in rows:
                    by_p[p]["pairs"][(c1, c2)] = (w, g)

                deltas, kish_n = build_decayed_residuals(
                    [by_p[p] for p in patches], rework_cutoffs, 0.87, 450.0 / (hm[r1][r2] ** 2), is_enemy
                )
                for (c1, c2), d in deltas.items():
                    target[(r1, c1, r2, c2)] = d
                    k_target[(r1, c1, r2, c2)] = kish_n[(c1, c2)]

                if (not is_enemy) or (r1 == r2):
                    for cid in {c1 for (c1, _) in deltas}:
                        if (r1, cid) not in champions:
                            continue
                        pts, w_sum = [], 0.0
                        for (c1, c2), d in deltas.items():
                            if c1 == cid and (r2, c2) in champions:
                                pr2 = champions[(r2, c2)].play_rate
                                pts.append((pr2, d, max(1.0, kish_n[(c1, c2)])))
                                w_sum += pr2
                        if w_sum > 0:
                            mean_d = sum(w * d for w, d, _ in pts) / w_sum
                            v_obs = sum((w / w_sum) * ((d - mean_d) ** 2) for w, d, _ in pts)
                            v_noise = sum((w / w_sum) * (4.0 / n) for w, _, n in pts)
                            spread = math.sqrt(max(0.0, v_obs - v_noise))
                            if is_enemy:
                                champions[(r1, cid)].matchup_spread = spread
                            else:
                                champions[(r1, cid)].synergy_spread[r2] = spread

    pick_totals, ban_counts = defaultdict(int), defaultdict(int)
    for st, b_cid, r, p_cid, cnt in con.execute(
        f"SELECT is_same_team, banned_cid, role, picked_cid, count FROM ban_pick_stats WHERE patch IN ({q_marks})",
        patches,
    ):
        ban_counts[(st, b_cid, r, p_cid)] += cnt
        pick_totals[(st, r, p_cid)] += cnt
    ban_likes = {
        (st, b_cid, r, p_cid): (cnt + 1.0) / (pick_totals[(st, r, p_cid)] / 5.0 + 60.0)
        for (st, b_cid, r, p_cid), cnt in ban_counts.items()
    }

    setup_accum = defaultdict(lambda: [0, 0])
    for role, cid, ecid, ks, sub, sp, w, g in con.execute(
        f"SELECT role, cid, enemy_cid, keystone_id, substyle_id, spell_pair, wins, games FROM setup_stats WHERE patch IN ({q_marks})",
        patches,
    ):
        setup_accum[(role, cid, ecid, ks, sub, sp)][0] += w
        setup_accum[(role, cid, ecid, ks, sub, sp)][1] += g
    setup_table = defaultdict(list)
    for (role, cid, ecid, ks, sub, sp), (w, g) in setup_accum.items():
        if g >= 10 and (role, cid) in champions:
            base_ref = marginal_wr_map.get((role, cid), champions[(role, cid)].base_wr)
            shrunk_z = (g / (g + 80.0)) * (logit((w + 5.0) / (g + 10.0)) - logit(base_ref))
            setup_table[(role, cid, ecid)].append(
                SetupOption(ks, rune_names.get(ks, f"Rune {ks}"), sub, rune_names.get(sub, f"Tree {sub}"), sp, shrunk_z, g)
            )

    con.close()

    if not excluded and fp is not None:
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump({
                    "fp": fp,
                    "champions": champions,
                    "pre_ctr": pre_ctr,
                    "pre_syn": pre_syn,
                    "ctr_kish": ctr_kish,
                    "syn_kish": syn_kish,
                    "ban_likes": ban_likes,
                    "setup_table": dict(setup_table),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass

    return DraftScoringEngine(champions, pre_ctr, pre_syn, ctr_kish, syn_kish, ban_likes, setup_table, personal, onnx_net)


def build_demo_engine(personal: PersonalProfile = None, onnx_net=None) -> DraftScoringEngine:
    """Sandbox showcasing OTP de-biasing, LCB uncertainty, soft ally hovers, & Ban Inference."""
    demo_champs = {
        ("MID", 103): ChampProfile(103, "Ahri", "MID", 0.518, 0.11, np.array([0.1, 0.9, 0.2, 0.6, 0.8, 0.5]), np.array([0.85, 0.1, 0.85, 0.05, 0.2, 0.6, 0.15, 0.8, 0.8, 0.5]), 2200, np.array([0.52, 0.52, 0.51, 0.50]), 0.01, {}, {}, 0.45, 0.01, 0.03, 4.0, {"JGL": 0.025, "SUP": 0.02}, 0.044),
        ("MID", 136): ChampProfile(136, "Aurelion Sol", "MID", 0.522, 0.06, np.array([0.05, 1.0, 0.2, 0.5, 1.0, 1.0]), np.array([0.85, 0.05, 0.9, 0.05, 0.2, 0.5, 0.05, 0.5, 1.0, 1.0]), 1400, np.array([0.44, 0.50, 0.54, 0.57]), 0.0, {}, {}, 0.35, 0.02, 0.04, 4.0, {"JGL": 0.04, "SUP": 0.03}, 0.065),
        ("MID", 61):  ChampProfile(61, "Orianna", "MID", 0.509, 0.08, np.array([0.1, 0.95, 0.2, 0.75, 0.9, 0.8]), np.array([0.81, 0.1, 0.9, 0.0, 0.2, 0.75, 0.3, 0.3, 0.9, 0.8]), 1800, np.array([0.49, 0.51, 0.52, 0.53]), 0.01, {}, {}, 0.60, 0.01, 0.04, 4.0, {"JGL": 0.05, "SUP": 0.02}, 0.048),
        ("MID", 268): ChampProfile(268, "Azir (OTP/Low-N)", "MID", 0.536, 0.012, np.array([0.1, 0.9, 0.2, 0.6, 0.9, 0.9]), np.array([0.81, 0.1, 0.9, 0.0, 0.2, 0.6, 0.05, 0.7, 0.9, 0.9]), 35, np.array([0.46, 0.50, 0.53, 0.56]), 0.0, {}, {}, 0.95, 0.09, 0.09, 4.0, {"JGL": 0.06, "SUP": 0.04}, 0.075),
        ("MID", 157): ChampProfile(157, "Yasuo", "MID", 0.508, 0.09, np.array([0.95, 0.1, 0.3, 0.4, 0.8, 0.65]), np.array([0.27, 0.9, 0.1, 0.0, 0.3, 0.4, 0.1, 0.9, 0.8, 0.65]), 1900, np.array([0.51, 0.50, 0.50, 0.49]), -0.01, {}, {}, 0.95, 0.07, 0.08, 4.0, {"JGL": 0.085, "SUP": 0.06}, 0.088),
        ("MID", 800): ChampProfile(800, "Mel", "MID", 0.495, 0.04, np.array([0.05, 0.95, 0.15, 0.5, 0.85, 0.7]), np.array([0.85, 0.05, 0.95, 0.0, 0.15, 0.5, 0.1, 0.3, 0.85, 0.7]), 900),
        ("MID", 79):  ChampProfile(79, "Gragas", "MID", 0.506, 0.04, np.array([0.1, 0.85, 0.65, 0.85, 0.7, 0.55]), np.array([0.20, 0.1, 0.9, 0.0, 0.7, 0.85, 0.3, 0.65, 0.7, 0.55]), 800),
        ("TOP", 79):  ChampProfile(79, "Gragas", "TOP", 0.508, 0.05, np.array([0.1, 0.85, 0.7, 0.85, 0.7, 0.55]), np.array([0.20, 0.1, 0.9, 0.0, 0.7, 0.85, 0.3, 0.65, 0.7, 0.55]), 900),
        ("JGL", 79):  ChampProfile(79, "Gragas", "JGL", 0.504, 0.04, np.array([0.1, 0.85, 0.65, 0.85, 0.6, 0.55]), np.array([0.20, 0.1, 0.9, 0.0, 0.7, 0.85, 0.3, 0.65, 0.6, 0.55]), 700),
        ("TOP", 516): ChampProfile(516, "Ornn", "TOP", 0.517, 0.07, np.array([0.2, 0.25, 1.0, 0.95, 0.5, 0.95]), np.array([0.27, 0.3, 0.7, 0.0, 1.0, 0.95, 0.1, 0.35, 0.5, 0.95]), 1500),
        ("JGL", 102): ChampProfile(102, "Shyvana", "JGL", 0.515, 0.08, np.array([0.65, 0.45, 0.65, 0.3, 0.7, 0.7]), np.array([0.20, 0.55, 0.4, 0.05, 0.65, 0.3, 0.25, 0.55, 0.7, 0.7]), 1600),
        ("SUP", 111): ChampProfile(111, "Nautilus", "SUP", 0.508, 0.12, np.array([0.1, 0.25, 0.85, 1.0, 0.1, 0.4]), np.array([0.27, 0.2, 0.8, 0.0, 0.85, 1.0, 0.2, 0.45, 0.1, 0.4]), 2100),
        ("BOT", 222): ChampProfile(222, "Jinx", "BOT", 0.519, 0.14, np.array([0.95, 0.05, 0.1, 0.25, 0.85, 0.9]), np.array([0.85, 0.95, 0.05, 0.0, 0.1, 0.25, 0.05, 0.4, 0.85, 0.9]), 2600),
    }
    demo_ctr = {("MID", 103, "MID", 800): -0.17, ("MID", 157, "TOP", 516): -0.14, ("MID", 136, "TOP", 516): 0.14}
    demo_syn = {("MID", 61, "JGL", 102): 0.10, ("MID", 157, "SUP", 111): 0.11, ("MID", 136, "SUP", 111): 0.07}
    demo_bans = {(1, 54, "MID", 157): 0.22}
    demo_setups = {
        ("MID", 136, 0): [SetupOption(8229, "Arcane Comet", 8300, "Inspiration", "4-12", 0.03, 500)],
        ("MID", 103, 0): [SetupOption(8112, "Electrocute", 8200, "Sorcery", "4-14", 0.02, 500)],
        ("MID", 61, 0):  [SetupOption(8214, "Summon Aery", 8300, "Inspiration", "4-12", 0.02, 400)],
        ("MID", 268, 0): [SetupOption(8010, "Conqueror", 8200, "Sorcery", "4-12", 0.02, 80)],
        ("MID", 157, 0): [SetupOption(8008, "Lethal Tempo", 8400, "Resolve", "4-14", 0.01, 400)],
    }
    pers = personal or PersonalProfile(
        mastery_points={103: 65000, 136: 48000, 157: 120000, 268: 500},
        mastery_logit={103: 0.05, 136: 0.04, 157: 0.06, 268: -0.08},
        recent_games_45d={136: 18, 103: 12, 157: 8},
        pool_ids={103, 136, 61, 157, 268},
        duration_slope=0.06,
    )
    return DraftScoringEngine(demo_champs, demo_ctr, demo_syn, {}, {}, demo_bans, demo_setups, pers, onnx_net)