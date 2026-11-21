import random, sqlite3, sys
from collections import Counter
import numpy as np

from core.scoring_engine import ROLES, MATCHUP_HEATMAP, SYNERGY_HEATMAP, DraftState, logit, sigmoid
from data.db_loader import load_engine_from_db


def _patch_key(p: str):
    return tuple(int(x) for x in p.split(".") if x.isdigit())


def per_sample_logloss(y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    p = np.clip(p_pred, 1e-5, 1.0 - 1e-5)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    p = np.clip(p_pred, 1e-5, 1.0 - 1e-5)
    ll = float(np.mean(per_sample_logloss(y_true, p)))
    brier = float(np.mean((p - y_true) ** 2))
    acc = float(np.mean((p >= 0.5) == y_true))

    order = np.argsort(p)
    y_sorted = y_true[order]
    n_pos = max(1, int(y_true.sum()))
    n_neg = max(1, len(y_true) - n_pos)
    auc = float((np.arange(1, len(y_true) + 1)[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(p)) * abs(float(p[m].mean()) - float(y_true[m].mean()))
    return {"Acc": round(acc * 100, 2), "LogLoss": round(ll, 4), "Brier": round(brier, 4), "AUC": round(auc, 4), "ECE": round(float(ece), 4)}


def fit_mixed_logreg(
    X: np.ndarray,
    y: np.ndarray,
    n_sparse: int = 0,
    l2_sparse: float = 350.0,
    l2_cont: float = 25.0,
    sample_weights: np.ndarray = None,
    iters: int = 6,
):
    """
    Exact Newton-IRLS L2 logistic regression with decoupled regularization:
      - Columns [:n_sparse] stay unstandardized with penalty l2_sparse
      - Columns [n_sparse:] are z-scored (mean 0, std 1) with penalty l2_cont
    """
    n, d = X.shape
    mu = np.zeros(d, dtype=np.float64)
    sd = np.ones(d, dtype=np.float64)
    if n_sparse < d:
        mu[n_sparse:] = X[:, n_sparse:].mean(axis=0)
        sd[n_sparse:] = np.maximum(X[:, n_sparse:].std(axis=0), 1e-6)
    Xs = (X - mu) / sd

    sw = np.ones(n, dtype=np.float64) if sample_weights is None else sample_weights.astype(np.float64)
    Xb = np.hstack([np.ones((n, 1), dtype=np.float64), Xs])
    w = np.zeros(d + 1, dtype=np.float64)

    reg = np.full(d + 1, l2_sparse, dtype=np.float64)
    reg[0] = 1e-4
    if n_sparse < d:
        reg[1 + n_sparse:] = l2_cont

    for _ in range(iters):
        z = np.clip(Xb @ w, -15.0, 15.0)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(p * (1.0 - p), 1e-6, 0.25) * sw
        grad = Xb.T @ (sw * (p - y)) + reg * w
        H = (Xb.T * W) @ Xb
        H[np.diag_indices_from(H)] += reg
        w -= np.linalg.solve(H, grad)
    return w, mu, sd


def predict_logreg(X: np.ndarray, model_tuple) -> np.ndarray:
    w, mu, sd = model_tuple
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), Xs])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -15.0, 15.0)))


def _load_games_for_patches(con: sqlite3.Connection, patches: list[str]):
    q = ",".join("?" * len(patches))
    matches = con.execute(
        f"SELECT match_id, patch, elo_tier, blue_win FROM matches WHERE patch IN ({q})", patches
    ).fetchall()
    parts_map = {}
    for mid, tid, role, cid, ks, s1, s2 in con.execute(f"""
        SELECT p.match_id, p.team_id, p.role, p.cid, p.keystone_id, p.spell1_id, p.spell2_id
        FROM match_participants p JOIN matches m ON p.match_id = m.match_id WHERE m.patch IN ({q})
    """, patches):
        sp = f"{min(s1, s2)}-{max(s1, s2)}"
        parts_map.setdefault(mid, []).append((tid, role, cid, ks, sp))

    games = []
    for mid, patch, elo, bwin in matches:
        plist = parts_map.get(mid, [])
        if len(plist) != 10:
            continue
        blue, red, b_setups, r_setups = {}, {}, {}, {}
        for tid, role, cid, ks, sp in plist:
            if tid == 100:
                blue[role] = cid; b_setups[role] = (ks, sp)
            else:
                red[role] = cid; r_setups[role] = (ks, sp)
        if len(blue) == 5 and len(red) == 5:
            games.append({
                "patch": patch, "elo": elo, "y": float(bwin),
                "blue": blue, "red": red, "b_setups": b_setups, "r_setups": r_setups,
            })
    return games


def _team_comp_features(team: dict, enemy: dict, champions: dict) -> np.ndarray:
    mv = sum((champions[(r, c)].mech_vec for r, c in team.items() if (r, c) in champions), np.zeros(10)) / 5.0
    tv = sum((champions[(r, c)].comp_vec for r, c in team.items() if (r, c) in champions), np.zeros(6))
    ev = sum((champions[(r, c)].comp_vec for r, c in enemy.items() if (r, c) in champions), np.zeros(6))
    tot_d = tv[0] + tv[1] + 1e-6
    phys_s, mag_s = tv[0] / tot_d, tv[1] / tot_d
    e_tank = min(2.5, ev[2])
    mono_phys = max(0.0, phys_s - 0.72) * (0.5 + e_tank)
    mono_mag = max(0.0, mag_s - 0.72) * (0.5 + e_tank)
    low_cc = max(0.0, 1.2 - tv[3])
    no_front = max(0.0, 0.8 - tv[2])
    return np.concatenate([mv, np.array([mono_phys, mono_mag, low_cc, no_front])])


def _team_comp_penalty(team: dict, enemy: dict, champions: dict) -> float:
    f = _team_comp_features(team, enemy, champions)
    return float(-(0.45 * f[10] + 0.45 * f[11] + 0.09 * f[12] + 0.07 * f[13]))


def _extract_phase1_features(engine, games):
    """6 Live Layer-1 features: [z_base(Joint BT), z_lane, z_cross, z_syn, z_comp, z_elo_side]."""
    y_list, feats, onnx_batch = [], [], []
    for g in games:
        blue, red, elo = g["blue"], g["red"], g["elo"]
        z_base = sum(logit(engine.champions[(r, c)].base_wr) for r, c in blue.items() if (r, c) in engine.champions) \
               - sum(logit(engine.champions[(r, c)].base_wr) for r, c in red.items() if (r, c) in engine.champions)
        z_lane = sum(engine.precomputed_matchups.get((r, blue[r], r, red[r]), 0.0) for r in ROLES)
        z_cross = sum(
            MATCHUP_HEATMAP[r1][r2] * engine.precomputed_matchups.get((r1, blue[r1], r2, red[r2]), 0.0)
            for r1 in ROLES for r2 in ROLES if r1 != r2
        )
        z_syn = sum(
            SYNERGY_HEATMAP[ROLES[i]][ROLES[j]] * engine.precomputed_synergies.get((ROLES[i], blue[ROLES[i]], ROLES[j], blue[ROLES[j]]), 0.0)
            for i in range(5) for j in range(i + 1, 5)
        ) - sum(
            SYNERGY_HEATMAP[ROLES[i]][ROLES[j]] * engine.precomputed_synergies.get((ROLES[i], red[ROLES[i]], ROLES[j], red[ROLES[j]]), 0.0)
            for i in range(5) for j in range(i + 1, 5)
        )
        z_comp = _team_comp_penalty(blue, red, engine.champions) - _team_comp_penalty(red, blue, engine.champions)
        z_es = sum(
            engine.champions[(r, c)].elo_wr_delta.get(elo, 0.0) + 0.5 * engine.champions[(r, c)].blue_side_delta
            for r, c in blue.items() if (r, c) in engine.champions
        ) - sum(
            engine.champions[(r, c)].elo_wr_delta.get(elo, 0.0) - 0.5 * engine.champions[(r, c)].blue_side_delta
            for r, c in red.items() if (r, c) in engine.champions
        )
        y_list.append(g["y"])
        feats.append([z_base, z_lane, z_cross, z_syn, z_comp, z_es])
        onnx_batch.append({"blue": blue, "red": red, "elo": elo, "l1_prior": 0.65 * z_base + 0.35 * z_lane})
    return np.array(y_list, dtype=np.float64), np.array(feats, dtype=np.float64), onnx_batch


def build_joint_design_matrix(games, champions, vocab=None, min_count: int = 25, include_setups: bool = True):
    """Builds sparse (+1 Blue, -1 Red) indicators + 14 continuous mechanics/comp features."""
    if vocab is None:
        counts = Counter()
        for g in games:
            for side_c, side_s in ((g["blue"], g["b_setups"]), (g["red"], g["r_setups"])):
                for r, c in side_c.items():
                    counts[("C", r, c)] += 1
                    if include_setups:
                        ks, sp = side_s[r]
                        counts[("K", r, ks)] += 1
                        counts[("S", r, sp)] += 1
        vocab = {k: i for i, k in enumerate(k for k, cnt in counts.items() if cnt >= min_count)}

    n_sparse = len(vocab)
    X = np.zeros((len(games), n_sparse + 14), dtype=np.float64)
    y = np.zeros(len(games), dtype=np.float64)

    for i, g in enumerate(games):
        y[i] = g["y"]
        for r, c in g["blue"].items():
            if ("C", r, c) in vocab:
                X[i, vocab[("C", r, c)]] += 1.0
            if include_setups:
                ks, sp = g["b_setups"][r]
                if ("K", r, ks) in vocab:
                    X[i, vocab[("K", r, ks)]] += 1.0
                if ("S", r, sp) in vocab:
                    X[i, vocab[("S", r, sp)]] += 1.0
        for r, c in g["red"].items():
            if ("C", r, c) in vocab:
                X[i, vocab[("C", r, c)]] -= 1.0
            if include_setups:
                ks, sp = g["r_setups"][r]
                if ("K", r, ks) in vocab:
                    X[i, vocab[("K", r, ks)]] -= 1.0
                if ("S", r, sp) in vocab:
                    X[i, vocab[("S", r, sp)]] -= 1.0
        X[i, n_sparse:] = _team_comp_features(g["blue"], g["red"], champions) - _team_comp_features(g["red"], g["blue"], champions)
    return X, y, vocab, n_sparse


def _patch_weights(games, patch_decay: float = 0.88) -> np.ndarray:
    patches_sorted = sorted({g["patch"] for g in games}, key=_patch_key, reverse=True)
    p_age = {p: idx for idx, p in enumerate(patches_sorted)}
    return np.array([patch_decay ** p_age[g["patch"]] for g in games], dtype=np.float64)


def tune_and_predict_joint(
    train_games, cal_train, cal_games, eval_games, cal_champions, champions,
    include_setups: bool = True, patch_decay: float = 0.88,
):
    """Tunes sparse L2 strictly on cal_patch across [150..2000], then fits on train_games -> eval_games."""
    best_l2, best_ll = 500.0, 9.0
    if len(cal_train) >= 1000 and len(cal_games) >= 500:
        X_ct, y_ct, v_c, ns_c = build_joint_design_matrix(cal_train, cal_champions, include_setups=include_setups)
        X_cv, y_cv, _, _ = build_joint_design_matrix(cal_games, cal_champions, vocab=v_c, include_setups=include_setups)
        sw_ct = _patch_weights(cal_train, patch_decay)
        for cand in (150.0, 350.0, 700.0, 1200.0, 2000.0):
            m = fit_mixed_logreg(X_ct, y_ct, n_sparse=ns_c, l2_sparse=cand, l2_cont=25.0, sample_weights=sw_ct, iters=5)
            ll = float(np.mean(per_sample_logloss(y_cv, predict_logreg(X_cv, m))))
            if ll < best_ll:
                best_ll, best_l2 = ll, cand

    X_tr, y_tr, vocab, n_sparse = build_joint_design_matrix(train_games, champions, include_setups=include_setups)
    X_ev, y_ev, _, _ = build_joint_design_matrix(eval_games, champions, vocab=vocab, include_setups=include_setups)
    sw_tr = _patch_weights(train_games, patch_decay)
    model = fit_mixed_logreg(X_tr, y_tr, n_sparse=n_sparse, l2_sparse=best_l2, l2_cont=25.0, sample_weights=sw_tr, iters=6)
    return predict_logreg(X_ev, model), y_ev, best_l2, len(vocab)


def _print_sig_delta(label: str, y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray):
    diff = per_sample_logloss(y, p_old) - per_sample_logloss(y, p_new)
    mean_gain = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(max(1, len(diff))))
    z_stat = mean_gain / max(se, 1e-9)
    verdict = "SIGNIFICANT" if z_stat >= 1.96 else ("WORSE" if z_stat <= -1.96 else "inconclusive")
    print(f"  ΔLogLoss ({label:<32}): {mean_gain:+.5f}  (z={z_stat:+.2f} -> {verdict})")


def _print_table(title: str, rows: list[tuple[str, dict]]):
    print(f"\n=== {title} ===")
    print(f"{'Model / Benchmark':<42} | {'Acc%':>6} | {'LogLoss':>7} | {'Brier':>6} | {'AUC':>6} | {'ECE':>6}")
    print("-" * 86)
    for name, m in rows:
        print(f"{name:<42} | {m['Acc']:>6.2f} | {m['LogLoss']:>7.4f} | {m['Brier']:>6.4f} | {m['AUC']:>6.4f} | {m['ECE']:>6.4f}")
    print("-" * 86)
    print(f"{'Published Ref: DraftGap (Apr 2026)':<42} | {'54.66':>6} | {'0.6869':>7} | {'0.2469':>6} | {'~0.56':>6} | {'~0.010':>6}")
    print(f"{'Published Ref: LoLDraftAI (2026, ~3M)':<42} | {'55.90':>6} | {'0.6824':>7} | {'0.2447':>6} | {'~0.58':>6} | {'~0.009':>6}")


def run_walk_forward_ablations(db_path="data/patch_matrix.db", target_patch: str = None):
    con = sqlite3.connect(db_path, timeout=30.0)
    patch_counts = dict(con.execute("SELECT patch, COUNT(*) FROM matches GROUP BY patch").fetchall())
    patches = sorted([p for p, cnt in patch_counts.items() if cnt >= 200], key=_patch_key)
    if len(patches) < 2:
        print(f"[Walk-Forward] Need >= 2 patches with 200+ matches. Found: {patch_counts}")
        con.close()
        return

    if target_patch and target_patch in patch_counts:
        test_patch = target_patch
    else:
        big = [p for p in patches[1:] if patch_counts[p] >= 3000]
        test_patch = big[-1] if big else patches[-1]

    test_idx = patches.index(test_patch)
    train_patches = [p for p in patches if _patch_key(p) < _patch_key(test_patch)]
    excluded_for_test = {p for p in patch_counts if _patch_key(p) >= _patch_key(test_patch)}

    train_games = _load_games_for_patches(con, train_patches)
    test_games = _load_games_for_patches(con, [test_patch])
    blue_prior = float(np.mean([g["y"] for g in train_games])) if train_games else 0.5
    con.close()

    # 1. Calibrate Live Layer-1 weights strictly on cal_patch (patches[test_idx - 1])
    cal_patch = patches[test_idx - 1]
    cal_games = [g for g in train_games if g["patch"] == cal_patch]
    cal_train = [g for g in train_games if g["patch"] != cal_patch]
    cal_engine = load_engine_from_db(db_path, exclude_patches=excluded_for_test | {cal_patch})
    y_c, X_c, _ = _extract_phase1_features(cal_engine, cal_games)
    m_A = fit_mixed_logreg(X_c[:, [0, 1]], y_c, n_sparse=0, l2_cont=3.0)
    m_B = fit_mixed_logreg(X_c[:, [0, 1, 2, 3, 4]], y_c, n_sparse=0, l2_cont=5.0)
    m_D = fit_mixed_logreg(X_c[:, [0, 1, 2, 3, 4, 5]], y_c, n_sparse=0, l2_cont=5.0)

    # 2. STRICT FUTURE-PATCH HOLDOUT
    engine = load_engine_from_db(db_path, exclude_patches=excluded_for_test)
    y_arr, X_t, onnx_batch = _extract_phase1_features(engine, test_games)
    p_0 = np.full_like(y_arr, blue_prior)
    p_A = predict_logreg(X_t[:, [0, 1]], m_A)
    p_B = predict_logreg(X_t[:, [0, 1, 2, 3, 4]], m_B)
    p_D = predict_logreg(X_t[:, [0, 1, 2, 3, 4, 5]], m_D)

    p_E1, _, l2_e1, nf_e1 = tune_and_predict_joint(
        train_games, cal_train, cal_games, test_games, cal_engine.champions, engine.champions, include_setups=False
    )
    p_E2, _, l2_e2, nf_e2 = tune_and_predict_joint(
        train_games, cal_train, cal_games, test_games, cal_engine.champions, engine.champions, include_setups=True
    )

    rows = [
        ("Ref: Naive Side Prior (No Draft)", compute_metrics(y_arr, p_0)),
        ("Live L1-A (Joint BT + Direct Lane)", compute_metrics(y_arr, p_A)),
        ("Live L1-B (+ Heatmap SVD & Comp)", compute_metrics(y_arr, p_B)),
        ("Live L1-D (+ Elo & Side Cond.)", compute_metrics(y_arr, p_D)),
        ("Baseline E1 (Pre-Game Champs+Mech)", compute_metrics(y_arr, p_E1)),
        ("Baseline E2 (+ Runes & Spells)", compute_metrics(y_arr, p_E2)),
    ]
      
    p_NN_live = p_NN_e2 = nn_z_pre = None
    if engine.onnx_net and engine.onnx_net.active:
        out_pre = engine.onnx_net.predict_batch(onnx_batch, engine.champions)
        nn_z_pre = out_pre["mean_win_logit"].astype(np.float64)
        p_NN_solo = np.array([sigmoid(float(z)) for z in nn_z_pre])
        p_NN_live = np.array([sigmoid(0.80 * nn_z_pre[i] + 0.20 * logit(p_D[i])) for i in range(len(y_arr))])
        p_NN_e2 = np.array([sigmoid(0.70 * nn_z_pre[i] + 0.30 * logit(p_E2[i])) for i in range(len(y_arr))])

        rows.append(("Layer 2 (Standalone 5x ONNX v3)", compute_metrics(y_arr, p_NN_solo)))
        rows.append(("Live Prod (0.80 ONNX v3 + 0.20 L1)", compute_metrics(y_arr, p_NN_live)))
        rows.append(("Post-Lock (0.70 ONNX v3 + 0.30 E2)", compute_metrics(y_arr, p_NN_e2)))

    _print_table(f"STRICT FUTURE-PATCH HOLDOUT (Train: {len(train_games):,} <{test_patch} | Test: {test_patch}, n={len(y_arr):,})", rows)
    print(f"Tuned on Patch {cal_patch}: E1 ({nf_e1} feats, l2={l2_e1:.0f}) | E2 ({nf_e2} feats, l2={l2_e2:.0f})")
    print("Paired Significance Checks (positive ΔLogLoss = upgrade beats baseline):")
    _print_sig_delta("Live L1-D (Joint) vs Naive", y_arr, p_0, p_D)
    _print_sig_delta("E2 (+Runes/Spells) vs E1 (Draft)", y_arr, p_E1, p_E2)
    if p_NN_live is not None:
        _print_sig_delta("Live Prod (ONNX+L1) vs Live L1-D", y_arr, p_D, p_NN_live)
        _print_sig_delta("Post-Lock (ONNX+E2) vs E2", y_arr, p_E2, p_NN_e2)

    # 3. 5-FOLD LEAK-FREE SAME-PATCH CV ON ALL GAMES OF test_patch
    if len(test_games) >= 1500:
        perm = list(range(len(test_games)))
        random.Random(42).shuffle(perm)
        shuffled = [test_games[i] for i in perm]
        nn_shuf = nn_z_pre[perm] if nn_z_pre is not None else None

        X_old, y_old, vocab_cv, ns_cv = build_joint_design_matrix(train_games, engine.champions, include_setups=True)
        X_cur, y_cur, _, _ = build_joint_design_matrix(shuffled, engine.champions, vocab=vocab_cv, include_setups=True)
        sw_old = _patch_weights(train_games, 0.85) * 0.85
        sw_cur = np.ones(len(shuffled), dtype=np.float64)

        n_cur = len(shuffled)
        oof_e2 = np.zeros(n_cur, dtype=np.float64)
        folds = np.array_split(np.arange(n_cur), 5)
        for val_idx in folds:
            mask = np.ones(n_cur, dtype=bool)
            mask[val_idx] = False
            X_f = np.vstack([X_old, X_cur[mask]])
            y_f = np.concatenate([y_old, y_cur[mask]])
            sw_f = np.concatenate([sw_old, sw_cur[mask]])
            m_f = fit_mixed_logreg(X_f, y_f, n_sparse=ns_cv, l2_sparse=l2_e2, l2_cont=25.0, sample_weights=sw_f, iters=5)
            oof_e2[val_idx] = predict_logreg(X_cur[val_idx], m_f)

        p_sp_0 = np.full(n_cur, blue_prior)
        sp_rows = [
            ("Ref: Naive Side Prior", compute_metrics(y_cur, p_sp_0)),
            ("Baseline E2 (Same-Patch 5-Fold OOF)", compute_metrics(y_cur, oof_e2)),
        ]
        oof_stack = None
        if nn_shuf is not None:
            oof_stack = np.array([sigmoid(0.65 * nn_shuf[i] + 0.35 * logit(oof_e2[i])) for i in range(n_cur)])
            sp_rows.append(("Online Hybrid (Frozen ONNX + SamePatch E2)", compute_metrics(y_cur, oof_stack)))

        _print_table(f"LEAK-FREE 5-FOLD SAME-PATCH CV (Patch {test_patch}, All n={n_cur:,} OOF)", sp_rows)
        _print_sig_delta("Same-Patch E2 vs Naive", y_cur, p_sp_0, oof_e2)
        if oof_stack is not None:
            _print_sig_delta("Online Hybrid vs Same-Patch E2", y_cur, oof_e2, oof_stack)


def run_adversarial_draft_sim(n_drafts: int = 500, db_path="data/patch_matrix.db"):
    """Side-balanced 500 B1-R5 drafts: Expectimax+LCB vs Masked-Average Point-Estimate."""
    engine = load_engine_from_db(db_path)
    rng = random.Random(42)
    expmax_wrs = []
    pick_order = [("B", 0), ("R", 0), ("R", 1), ("B", 1), ("B", 2), ("R", 2), ("R", 3), ("B", 3), ("B", 4), ("R", 4)]

    for d_i in range(n_drafts):
        our_side = "B" if (d_i % 2 == 0) else "R"
        b_roles, r_roles = ROLES[:], ROLES[:]
        rng.shuffle(b_roles); rng.shuffle(r_roles)
        blue_locked, red_locked = {}, {}

        for side, idx in pick_order:
            role = b_roles[idx] if side == "B" else r_roles[idx]
            allies = blue_locked if side == "B" else red_locked
            enemies = red_locked if side == "B" else blue_locked
            is_ours = (side == our_side)
            st = DraftState(
                my_role=role, my_pick_index=idx, is_blue_side=(side == "B"),
                locked_allies=allies, enemy_locks=list(enemies.values()),
                mastery_mode="OFF",
                n_enemy_pool=7 if is_ours else 0,
                lcb_kappa=0.75 if is_ours else 0.0,
            )
            recs = engine.recommend(st, top_k=3)
            if recs:
                allies[role] = rng.choice(recs[:2])["champ_id"]

        if len(blue_locked) == 5 and len(red_locked) == 5:
            z_blue = sum(logit(engine.champions[(r, blue_locked[r])].base_wr) - logit(engine.champions[(r, red_locked[r])].base_wr) for r in ROLES) \
                   + sum(engine.precomputed_matchups.get((r1, blue_locked[r1], r2, red_locked[r2]), 0.0) for r1 in ROLES for r2 in ROLES) * 0.45
            p_blue = sigmoid(z_blue)
            expmax_wrs.append(p_blue if our_side == "B" else (1.0 - p_blue))

    if expmax_wrs:
        mean_wr = 100.0 * float(np.mean(expmax_wrs))
        print(f"\n=== Side-Balanced Adversarial Draft Simulation ({len(expmax_wrs)} B1-R5 Drafts) ===")
        print(f"Expectimax+LCB vs Masked Point-Estimate: {mean_wr:.2f}% ({mean_wr - 50.0:+.2f} pp)")


def print_personal_calibration(db_path="data/patch_matrix.db"):
    try:
        con = sqlite3.connect(db_path, timeout=30.0)
        rows = con.execute("SELECT followed_top2, pred_wr, actual_win FROM personal_calibration_log WHERE actual_win IN (0,1)").fetchall()
        con.close()
    except Exception:
        return
    if not rows:
        print("\n[Personal Calibration] No completed live games logged yet.")
        return
    arr = np.array(rows, dtype=np.float64)
    print(f"\n=== Personal Calibration Report (n={len(arr)} live games) ===")
    print("Overall:", compute_metrics(arr[:, 2], arr[:, 1]))
    foll, over = arr[arr[:, 0] == 1], arr[arr[:, 0] == 0]
    if len(foll):
        print(f"Followed #1/#2 ({len(foll)} games): Actual WR = {100*foll[:,2].mean():.1f}%")
    if len(over):
        print(f"Overrode Pick  ({len(over)} games): Actual WR = {100*over[:,2].mean():.1f}%")


if __name__ == "__main__":
    patch_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_walk_forward_ablations(target_patch=patch_arg)
    run_adversarial_draft_sim(n_drafts=500)
    print_personal_calibration()