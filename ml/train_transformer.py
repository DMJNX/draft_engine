import json, math, os, sqlite3, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.db_loader import load_engine_from_db
from ml.transformer_model import MECH_DIM, DraftEnsembleONNX, DraftSetTransformer

ROLE_TO_IDX = {"TOP": 0, "JGL": 1, "MID": 2, "BOT": 3, "SUP": 4}
ELO_TO_IDX = {"IRON": 0, "BRONZE": 0, "SILVER": 1, "GOLD": 1, "PLATINUM": 2, "EMERALD": 3, "DIAMOND": 4, "MASTER": 5}


def _patch_key(p: str):
    return tuple(int(x) for x in p.split(".") if x.isdigit())


def _eval_binary_metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0))), 1e-6, 1.0 - 1e-6)
    ll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    brier = float(np.mean((p - y) ** 2))
    acc = 100.0 * float(np.mean((p >= 0.5) == y))
    order = np.argsort(p)
    ys = y[order]
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, len(y) - n_pos)
    auc = float((np.arange(1, len(y) + 1)[ys == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(p)) * abs(float(p[m].mean()) - float(y[m].mean()))
    return {"Acc": acc, "LogLoss": ll, "Brier": brier, "AUC": auc, "ECE": float(ece)}


def load_training_dataset(db_path="data/patch_matrix.db", holdout_patch: str = None, n_folds: int = 5, patch_decay: float = 0.88):
    con = sqlite3.connect(db_path, timeout=30.0)
    patch_counts = dict(con.execute("SELECT patch, COUNT(*) FROM matches GROUP BY patch").fetchall())
    patches = sorted([p for p, cnt in patch_counts.items() if cnt >= 200], key=_patch_key)
    if not patches:
        con.close()
        raise RuntimeError("No patches with >=200 matches found.")

    if holdout_patch is not None and holdout_patch.lower() == "none":
        excluded = set()
        holdout_label = "NONE (Full Production)"
    else:
        if holdout_patch and holdout_patch in patch_counts:
            hp = holdout_patch
        else:
            big = [p for p in patches[1:] if patch_counts[p] >= 3000]
            hp = big[-1] if big else patches[-1]
        excluded = {p for p in patch_counts if _patch_key(p) >= _patch_key(hp)}
        holdout_label = f"Patch {hp} (100% Zero-Touch)"

    train_patches = sorted([p for p in patch_counts if p not in excluded], key=_patch_key, reverse=True)
    patch_age = {p: idx for idx, p in enumerate(train_patches)}  # 0 = latest training patch
    meta_cids = {r[0] for r in con.execute("SELECT cid FROM champ_meta")}

    q_tr = ",".join("?" * len(train_patches))
    matches = con.execute(f"""
        SELECT match_id, patch, elo_tier, duration_sec, blue_win,
               first_tower_blue, grubs_blue, grubs_red, soul_blue, soul_red, baron_blue, baron_red
        FROM matches WHERE patch IN ({q_tr}) AND duration_sec >= 900
    """, train_patches).fetchall()
    if len(matches) < 2000:
        con.close()
        raise RuntimeError(f"Need >=2,000 training matches; found {len(matches)}.")

    parts_by_mid, all_cids = defaultdict(list), set(meta_cids)
    for row in con.execute(f"""
        SELECT p.match_id, p.team_id, p.role, p.cid,
               p.gd15, p.xpd15, p.phys_share, p.mag_share, p.true_share, p.mit_dpm, p.heal_shield_pm
        FROM match_participants p JOIN matches m ON p.match_id = m.match_id
        WHERE m.patch IN ({q_tr}) AND m.duration_sec >= 900
    """, train_patches):
        parts_by_mid[row[0]].append(row[1:])
        all_cids.add(row[3])
    con.close()

    engine = load_engine_from_db(db_path, exclude_patches=excluded)
    cid_to_idx = {cid: i + 1 for i, cid in enumerate(sorted(all_cids))}

    n = len(matches)
    c_all = np.zeros((n, 10), dtype=np.int64)
    mv_all = np.zeros((n, 10, MECH_DIM), dtype=np.float32)
    r_all = np.tile(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64), (n, 1))
    t_all = np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64), (n, 1))
    s_all = np.zeros((n, 10), dtype=np.int64)
    elo_all = np.zeros(n, dtype=np.int64)
    pidx_all = np.zeros(n, dtype=np.int64)
    sw_all = np.zeros(n, dtype=np.float32)
    y_win = np.zeros(n, dtype=np.float32)
    y_tmp = np.zeros((n, 5, 2), dtype=np.float32)
    has_tmp = np.zeros(n, dtype=np.float32)
    y_db = np.zeros(n, dtype=np.int64)
    y_obj = np.zeros((n, 4), dtype=np.float32)
    y_pstat = np.zeros((n, 10, 5), dtype=np.float32)
    match_ids, patch_tags = [], []

    k = 0
    for mid, patch, elo, dur, bwin, ft, gb, gr, sb, sr, bb, br in matches:
        plist = parts_by_mid.get(mid)
        if not plist or len(plist) != 10:
            continue
        b_gd, r_gd = np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)
        b_xpd, r_xpd = np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)
        b_cnt = r_cnt = 0

        for tid, role, cid, gd, xpd, ph, mg, tr, mit, hs in plist:
            ri = ROLE_TO_IDX.get(role)
            if ri is None:
                continue
            slot = ri if tid == 100 else 5 + ri
            if tid == 100:
                b_cnt += 1
                b_gd[ri] = float(gd or 0.0); b_xpd[ri] = float(xpd or 0.0)
            else:
                r_cnt += 1
                r_gd[ri] = float(gd or 0.0); r_xpd[ri] = float(xpd or 0.0)

            c_all[k, slot] = cid_to_idx.get(cid, 0)
            prof = engine.champions.get((role, cid))
            if prof is not None:
                mv_all[k, slot] = prof.mech_vec.astype(np.float32)
            y_pstat[k, slot] = [
                float(ph or 0.0), float(mg or 0.0), float(tr or 0.0),
                min(3.0, float(mit or 0.0) / 2500.0),
                min(3.0, float(hs or 0.0) / 600.0),
            ]
        if b_cnt != 5 or r_cnt != 5:
            continue

        age = patch_age[patch]
        elo_all[k] = ELO_TO_IDX.get(elo, 3)
        pidx_all[k] = age
        sw_all[k] = float(patch_decay ** age)
        y_win[k] = float(bwin)
        y_tmp[k, :, 0] = b_gd - r_gd
        y_tmp[k, :, 1] = b_xpd - r_xpd
        has_tmp[k] = 1.0 if (np.abs(b_gd).sum() + np.abs(r_gd).sum() + np.abs(b_xpd).sum() > 1e-3) else 0.0

        dur_min = (dur or 1800) / 60.0
        y_db[k] = 0 if dur_min < 25 else (1 if dur_min < 30 else (2 if dur_min < 35 else 3))
        y_obj[k] = [
            float(np.clip(ft or 0, 0, 1)),
            float(np.clip(((gb or 0) - (gr or 0)) / 6.0, -1.0, 1.0)),
            float(np.clip((sb or 0) - (sr or 0), -1.0, 1.0)),
            float(math.tanh(((bb or 0) - (br or 0)) / 2.0)),
        ]
        match_ids.append(mid)
        patch_tags.append(patch)
        k += 1

    valid_t = has_tmp[:k] > 0.5
    if valid_t.any():
        gd_sd = max(1.0, float(np.std(y_tmp[:k, :, 0][valid_t])))
        xpd_sd = max(1.0, float(np.std(y_tmp[:k, :, 1][valid_t])))
        y_tmp[:k, :, 0] = np.clip(y_tmp[:k, :, 0] / gd_sd, -4.0, 4.0)
        y_tmp[:k, :, 1] = np.clip(y_tmp[:k, :, 1] / xpd_sd, -4.0, 4.0)

    rng = np.random.default_rng(42)
    fold_ids = np.zeros(k, dtype=np.int64)
    by_patch = defaultdict(list)
    for idx, p in enumerate(patch_tags):
        by_patch[p].append(idx)
    for idxs in by_patch.values():
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        for pos, sample_idx in enumerate(arr):
            fold_ids[sample_idx] = pos % n_folds

    tensors = (
        torch.from_numpy(c_all[:k]),
        torch.from_numpy(mv_all[:k]),
        torch.from_numpy(r_all[:k]),
        torch.from_numpy(t_all[:k]),
        torch.from_numpy(s_all[:k]),
        torch.from_numpy(elo_all[:k]),
        torch.from_numpy(pidx_all[:k]),
        torch.from_numpy(sw_all[:k]),
        torch.from_numpy(y_win[:k]),
        torch.from_numpy(y_tmp[:k]),
        torch.from_numpy(has_tmp[:k]),
        torch.from_numpy(y_db[:k]),
        torch.from_numpy(y_obj[:k]),
        torch.from_numpy(y_pstat[:k]),
    )
    print(f"[Phase 2 v4 Dataset] Train Pool: {k:,} matches (>=15m) across {len(train_patches)} patches | Holdout: {holdout_label}")
    return tensors, fold_ids, np.array(match_ids), np.array(patch_tags), train_patches, cid_to_idx


def _subset_ds(tensors, mask: np.ndarray) -> TensorDataset:
    idx = torch.from_numpy(np.nonzero(mask)[0])
    return TensorDataset(*[t[idx] for t in tensors])


def _fit_temperature(logits: np.ndarray, y: np.ndarray, sw: np.ndarray) -> float:
    best_t, best_ll = 1.0, 1e9
    for t in np.linspace(0.75, 1.50, 76):
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits / t, -15.0, 15.0))), 1e-6, 1.0 - 1e-6)
        ll = float(np.sum(sw * -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) / np.sum(sw))
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    return best_t


def _make_stage2_optimizer(model: DraftSetTransformer):
    win_no_decay, win_decay, bb_no_decay, bb_decay = [], [], [], []
    win_keys = ("rc_bias", "rc_patch_delta", "team_win_head", "lane_win_head", "dur_delta_head", "elo_side_bias", "dur_side_bias", "marg_gate", "role_ctr_w", "role_syn_w")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_win = any(k in name for k in win_keys)
        no_dec = (p.ndim == 1) or ("bias" in name) or ("norm" in name) or ("rc_patch_delta" in name) or ("role_ctr_w" in name) or ("role_syn_w" in name)
        if is_win and no_dec:
            win_no_decay.append(p)
        elif is_win:
            win_decay.append(p)
        elif no_dec:
            bb_no_decay.append(p)
        else:
            bb_decay.append(p)
    return torch.optim.AdamW([
        {"params": win_no_decay, "lr": 1.6e-3, "weight_decay": 0.0},
        {"params": win_decay, "lr": 1.2e-3, "weight_decay": 0.03},
        {"params": bb_no_decay, "lr": 4.5e-4, "weight_decay": 0.0},
        {"params": bb_decay, "lr": 4.5e-4, "weight_decay": 0.05},
    ])


def train_ensemble_and_export(
    db_path="data/patch_matrix.db",
    onnx_out="data/draft_transformer_ensemble.onnx",
    holdout_patch: str = None,
    n_models: int = 5,
    stage1_epochs: int = 8,
    stage2_epochs: int = 12,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2 v4 Two-Stage Trainer] Device: {device}")
    tensors, fold_ids, match_ids, patch_tags, train_patches, cid_to_idx = load_training_dataset(
        db_path, holdout_patch=holdout_patch, n_folds=n_models
    )
    n_champs = max(250, len(cid_to_idx) + 5)
    n_patches = len(train_patches)
    n_total = len(fold_ids)

    trained_models, temps = [], []
    oof_win = np.zeros(n_total, dtype=np.float32)
    oof_dur_win = np.zeros((n_total, 4), dtype=np.float32)
    oof_dur_cls = np.zeros((n_total, 4), dtype=np.float32)
    oof_tempo = np.zeros((n_total, 5, 2), dtype=np.float32)
    oof_obj = np.zeros((n_total, 4), dtype=np.float32)

    for fold in range(n_models):
        torch.manual_seed(400 + fold)
        np.random.seed(400 + fold)
        val_mask = (fold_ids == fold)
        train_loader = DataLoader(_subset_ds(tensors, ~val_mask), batch_size=384, shuffle=True)
        val_loader = DataLoader(_subset_ds(tensors, val_mask), batch_size=512, shuffle=False)

        model = DraftSetTransformer(
            num_champs=n_champs, num_patches=n_patches, d_model=96, nhead=4, num_layers=3, fm_dim=24, dropout=0.22
        ).to(device)
        zero_l1 = torch.zeros(1, device=device)

        # === STAGE 1: Dense Physics Pretraining (GD@15, XPD@15, Objectives, Duration, Stats — ZERO Win Loss) ===
        opt1 = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=0.03)
        tot1 = max(1, stage1_epochs * len(train_loader))
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=tot1, eta_min=1e-4)

        for _ in range(stage1_epochs):
            model.train()
            for batch in train_loader:
                c, mv, r, t, s, elo, pidx, sw, yw, ytmp, htmp, ydb, yobj, ypstat = [x.to(device) for x in batch]
                _, _, p_tmp, p_dcls, p_obj, p_stat = model(c, mv, r, t, s, elo, zero_l1.expand(c.size(0)), patch_idx=pidx)

                tmp_err = F.smooth_l1_loss(p_tmp, ytmp, reduction="none").mean(dim=(1, 2))
                loss_tmp = (tmp_err * htmp).sum() / htmp.sum().clamp_min(1.0)
                loss_dcls = F.cross_entropy(p_dcls, ydb)
                loss_obj = F.binary_cross_entropy_with_logits(p_obj[:, 0], yobj[:, 0]) + F.smooth_l1_loss(p_obj[:, 1:], yobj[:, 1:])
                loss_stat = F.smooth_l1_loss(p_stat, ypstat)

                loss = 0.60 * loss_tmp + 0.18 * loss_dcls + 0.30 * loss_obj + 0.25 * loss_stat + 0.25 * model.reg_loss()
                opt1.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt1.step(); sched1.step()

        # === STAGE 2: Patch-Delta Win & Scaling Fine-Tuning with EMA ===
        opt2 = _make_stage2_optimizer(model)
        tot2 = max(1, stage2_epochs * len(train_loader))
        warm2 = max(1, len(train_loader))

        def lr_lam(step):
            if step < warm2:
                return float(step + 1) / float(warm2)
            prog = float(step - warm2) / float(max(1, tot2 - warm2))
            return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lam)
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema_decay = 0.994
        best_loss, best_state = 1e9, None

        for ep in range(1, stage2_epochs + 1):
            model.train()
            for batch in train_loader:
                c, mv, r, t, s, elo, pidx, sw, yw, ytmp, htmp, ydb, yobj, ypstat = [x.to(device) for x in batch]
                if np.random.rand() < 0.15:
                    mask = torch.rand_like(c.float()) < 0.25
                    c = c.masked_fill(mask, 0)
                    mv = mv.masked_fill(mask.unsqueeze(-1), 0.0)

                w_log, p_dur, p_tmp, p_dcls, p_obj, p_stat = model(c, mv, r, t, s, elo, zero_l1.expand(c.size(0)), patch_idx=pidx)
                dur_sel = p_dur.gather(1, ydb.unsqueeze(1)).squeeze(1)

                sw_norm = sw / sw.mean().clamp_min(1e-6)
                loss_win = (F.binary_cross_entropy_with_logits(w_log, yw, reduction="none") * sw_norm).mean()
                loss_dur_w = (F.binary_cross_entropy_with_logits(dur_sel, yw, reduction="none") * sw_norm).mean()
                tmp_err = F.smooth_l1_loss(p_tmp, ytmp, reduction="none").mean(dim=(1, 2))
                loss_tmp = (tmp_err * htmp).sum() / htmp.sum().clamp_min(1.0)
                loss_dcls = F.cross_entropy(p_dcls, ydb)
                loss_obj = F.binary_cross_entropy_with_logits(p_obj[:, 0], yobj[:, 0]) + F.smooth_l1_loss(p_obj[:, 1:], yobj[:, 1:])
                loss_stat = F.smooth_l1_loss(p_stat, ypstat)

                loss = (
                    loss_win + 0.45 * loss_dur_w
                    + 0.18 * loss_tmp + 0.06 * loss_dcls + 0.10 * loss_obj + 0.06 * loss_stat
                    + model.reg_loss()
                )
                opt2.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step(); sched2.step()

                with torch.no_grad():
                    for k_name, v_cur in model.state_dict().items():
                        if v_cur.is_floating_point():
                            ema_state[k_name].mul_(ema_decay).add_(v_cur, alpha=1.0 - ema_decay)
                        else:
                            ema_state[k_name].copy_(v_cur)

            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state if ep >= 2 else live_state)
            model.eval()
            v_sum, w_tot = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    c, mv, r, t, s, elo, pidx, sw, yw, *_ = [x.to(device) for x in batch]
                    w0, *_ = model(c, mv, r, t, s, elo, zero_l1.expand(c.size(0)), patch_idx=pidx)
                    v_sum += float((F.binary_cross_entropy_with_logits(w0, yw, reduction="none") * sw).sum().item())
                    w_tot += float(sw.sum().item())
            v_loss = v_sum / max(1e-6, w_tot)
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(live_state)

        model.load_state_dict(best_state)
        model.eval()
        vl, vdw, vdc, vtmp, vobj, vy, vsw = [], [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                c, mv, r, t, s, elo, pidx, sw, yw, *_ = [x.to(device) for x in batch]
                w0, dw, tm, dc, ob, _ = model(c, mv, r, t, s, elo, zero_l1.expand(c.size(0)), patch_idx=pidx)
                vl.append(w0.cpu().numpy()); vdw.append(dw.cpu().numpy())
                vdc.append(torch.softmax(dc, dim=-1).cpu().numpy())
                vtmp.append(tm.cpu().numpy()); vobj.append(ob.cpu().numpy())
                vy.append(yw.cpu().numpy()); vsw.append(sw.cpu().numpy())

        l_np = np.concatenate(vl)
        y_np = np.concatenate(vy)
        sw_np = np.concatenate(vsw)
        t_opt = _fit_temperature(l_np, y_np, sw_np)

        val_idx = np.nonzero(val_mask)[0]
        oof_win[val_idx] = (l_np / t_opt).astype(np.float32)
        oof_dur_win[val_idx] = np.concatenate(vdw)
        oof_dur_cls[val_idx] = np.concatenate(vdc)
        oof_tempo[val_idx] = np.concatenate(vtmp)
        oof_obj[val_idx] = np.concatenate(vobj)
        temps.append(t_opt)
        print(f"Ensemble Fold {fold+1}/{n_models} | Best Two-Stage OOF LogLoss: {best_loss:.4f} | Temp: {t_opt:.2f}")

        model.cpu().eval()
        trained_models.append(model)

    y_all = tensors[8].numpy()
    m_all = _eval_binary_metrics(y_all, oof_win)
    print(f"  -> All-Train OOF (n={len(y_all):,}): Acc={m_all['Acc']:.2f}% | LogLoss={m_all['LogLoss']:.4f} | Brier={m_all['Brier']:.4f} | AUC={m_all['AUC']:.4f} | ECE={m_all['ECE']:.4f}")

    lp_mask = (patch_tags == train_patches[0])
    if lp_mask.sum() >= 1000:
        m_lp = _eval_binary_metrics(y_all[lp_mask], oof_win[lp_mask])
        print(f"  -> SAME-PATCH OOF Patch {train_patches[0]} (n={int(lp_mask.sum()):,}): Acc={m_lp['Acc']:.2f}% | LogLoss={m_lp['LogLoss']:.4f} | Brier={m_lp['Brier']:.4f} | AUC={m_lp['AUC']:.4f} | ECE={m_lp['ECE']:.4f}")

    ensemble = DraftEnsembleONNX(trained_models, temps=temps).eval()
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    with open("data/transformer_vocab.json", "w") as f:
        json.dump({"cid_to_idx": cid_to_idx}, f)

    torch.save({
        "state_dict": ensemble.state_dict(),
        "num_champs": n_champs,
        "num_patches": n_patches,
        "temps": temps,
        "train_patches": train_patches,
    }, "data/draft_transformer_ensemble.pt")

    np.savez_compressed(
        "data/v4_oof.npz",
        match_ids=match_ids,
        patch_tags=patch_tags,
        fold_ids=fold_ids,
        y_win=y_all,
        oof_win=oof_win,
        oof_dur_win=oof_dur_win,
        oof_dur_cls=oof_dur_cls,
        oof_tempo=oof_tempo,
        oof_obj=oof_obj,
    )

    dummy = (
        torch.zeros((1, 10), dtype=torch.long),
        torch.zeros((1, 10, MECH_DIM), dtype=torch.float32),
        torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long),
        torch.zeros((1, 10), dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0.0], dtype=torch.float32),
    )
    torch.onnx.export(
        ensemble, dummy, onnx_out,
        input_names=["champ_idx", "mech_vec", "role_idx", "team_idx", "setup_idx", "elo_idx", "l1_prior"],
        output_names=["mean_win_logit", "std_win_logit", "mean_duration_logits", "mean_dur_probs", "mean_tempo", "mean_obj"],
        dynamic_axes={k: {0: "batch"} for k in ["champ_idx", "mech_vec", "role_idx", "team_idx", "setup_idx", "elo_idx", "l1_prior"]},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported 5-Model v4 Two-Stage Ensemble to {onnx_out} & saved 19-dim OOF features to data/v4_oof.npz.")


if __name__ == "__main__":
    hp_arg = sys.argv[1] if len(sys.argv) > 1 else None
    train_ensemble_and_export(holdout_patch=hp_arg)