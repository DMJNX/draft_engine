import json, os
import numpy as np

from ml.stacker_runtime import StackerV2Runtime

ROLE_TO_IDX = {"TOP": 0, "JGL": 1, "MID": 2, "BOT": 3, "SUP": 4}
ELO_TO_IDX = {"IRON": 0, "BRONZE": 0, "SILVER": 1, "GOLD": 1, "PLATINUM": 2, "EMERALD": 3, "DIAMOND": 4, "MASTER": 5}
MECH_DIM = 10

# Game-length bucket weights [<25m, 25-30m, 30-35m, 35m+] used to reweight v4's scaling head for slower low-Elo games.
# High-Elo baseline matches Master/Diamond training distribution -> exact 0.0000 shift on Diamond/Master.
_HIGH_ELO_DUR_W = np.array([0.24, 0.34, 0.25, 0.17], dtype=np.float64)
_ELO_DUR_WEIGHTS = {
    "IRON":     np.array([0.12, 0.24, 0.30, 0.34], dtype=np.float64),
    "BRONZE":   np.array([0.13, 0.25, 0.30, 0.32], dtype=np.float64),
    "SILVER":   np.array([0.14, 0.26, 0.30, 0.30], dtype=np.float64),
    "GOLD":     np.array([0.16, 0.28, 0.29, 0.27], dtype=np.float64),
    "PLATINUM": np.array([0.19, 0.31, 0.28, 0.22], dtype=np.float64),
    "EMERALD":  np.array([0.21, 0.32, 0.27, 0.20], dtype=np.float64),
    "DIAMOND":  _HIGH_ELO_DUR_W,
    "MASTER":   _HIGH_ELO_DUR_W,
}

# Neutralizes Master+'s inverted 47.9% Blue / 52.1% Red matchmaking skew when scoring lower-rank lobbies.
_ELO_BLUE_SIDE_OFFSET = {
    "IRON": 0.085, "BRONZE": 0.085, "SILVER": 0.080, "GOLD": 0.075,
    "PLATINUM": 0.060, "EMERALD": 0.040, "DIAMOND": 0.0, "MASTER": 0.0,
}


def _setup_hash_pair(ks: int, sp: str, mod: int = 511) -> int:
    if not ks or ks <= 0:
        return 0
    try:
        a, b = [int(x) for x in str(sp).split("-")[:2]]
    except Exception:
        a, b = 0, 0
    return ((int(ks) * 31 + min(a, b) * 7 + max(a, b)) % mod) + 1


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def _np_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(pc / (1.0 - pc))


class ONNXValueNet:
    """
    Runs the 5-Model v4 Antisymmetric Ensemble on CPU, automatically fused with
    Stacker v2 (Tabular LGBM + Stacked LGBM + Calibrated ResMLP) when present.
    """
    def __init__(
        self,
        onnx_path="data/draft_transformer_ensemble.onnx",
        vocab_path="data/transformer_vocab.json",
        stacker_path="data/stacker_v2_bundle.pkl",
    ):
        self.active = False
        self.stacker_active = False
        self.session = None
        self.input_names = set()
        self.n_outputs = 0
        self.cid_to_idx = {}
        self.trained_elo_indices = {4, 5}  # Default: Diamond/Master until retrained on multi-Elo data
        self.stacker = None
        if not (os.path.exists(onnx_path) and os.path.exists(vocab_path)):
            return
        try:
            import onnxruntime as ort
            with open(vocab_path) as f:
                vdata = json.load(f)
                self.cid_to_idx = {int(k): int(v) for k, v in vdata["cid_to_idx"].items()}
                if "trained_elo_indices" in vdata:
                    self.trained_elo_indices = {int(x) for x in vdata["trained_elo_indices"]}
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.input_names = {inp.name for inp in self.session.get_inputs()}
            self.n_outputs = len(self.session.get_outputs())
            self.active = True
            if self.n_outputs >= 6 and os.path.exists(stacker_path):
                rt = StackerV2Runtime(stacker_path)
                if rt.active:
                    self.stacker = rt
                    self.stacker_active = True
        except Exception:
            self.active = False

    def _safe_elo_idx(self, elo_str: str) -> int:
        raw_idx = ELO_TO_IDX.get(elo_str, 3)
        if raw_idx in self.trained_elo_indices:
            return raw_idx
        return min(self.trained_elo_indices, key=lambda x: abs(x - raw_idx))

    def predict_batch(self, drafts: list[dict], champions: dict):
        if not self.active or not drafts:
            return None
        bsz = len(drafts)
        c_arr = np.zeros((bsz, 10), dtype=np.int64)
        m_arr = np.zeros((bsz, 10, MECH_DIM), dtype=np.float32)
        r_arr = np.tile(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64), (bsz, 1))
        t_arr = np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64), (bsz, 1))
        s_arr = np.zeros((bsz, 10), dtype=np.int64)
        e_arr = np.zeros((bsz,), dtype=np.int64)
        l1 = np.zeros((bsz,), dtype=np.float32)
        raw_elos = []

        for i, d in enumerate(drafts):
            b_setups = d.get("b_setups") or {}
            r_setups = d.get("r_setups") or {}
            for role, cid in d["blue"].items():
                ri = ROLE_TO_IDX[role]
                c_arr[i, ri] = self.cid_to_idx.get(cid, 0)
                if (role, cid) in champions:
                    m_arr[i, ri] = champions[(role, cid)].mech_vec.astype(np.float32)
                if role in b_setups:
                    s_arr[i, ri] = _setup_hash_pair(b_setups[role][0], b_setups[role][1])
            for role, cid in d["red"].items():
                ri = 5 + ROLE_TO_IDX[role]
                c_arr[i, ri] = self.cid_to_idx.get(cid, 0)
                if (role, cid) in champions:
                    m_arr[i, ri] = champions[(role, cid)].mech_vec.astype(np.float32)
                if role in r_setups:
                    s_arr[i, ri] = _setup_hash_pair(r_setups[role][0], r_setups[role][1])
            elo_str = str(d.get("elo", "EMERALD")).upper()
            raw_elos.append(elo_str)
            e_arr[i] = self._safe_elo_idx(elo_str)
            l1[i] = float(d.get("l1_prior", 0.0))

        full_feed = {
            "champ_idx": c_arr, "mech_vec": m_arr, "role_idx": r_arr, "team_idx": t_arr,
            "setup_idx": s_arr, "elo_idx": e_arr, "l1_prior": l1,
        }
        feed = {k: v for k, v in full_feed.items() if k in self.input_names}
        outs = self.session.run(None, feed)
        mean_w, std_w, dur = outs[0], outs[1], outs[2]
        n_filled = (c_arr > 0).sum(axis=1).astype(np.float64)

        if self.stacker_active and len(outs) >= 6:
            dur_probs, tempo, obj = outs[3], outs[4], outs[5]
            meta_z, fam_std = self.stacker.score_batch(
                c_arr, m_arr, e_arr, mean_w, dur, dur_probs, tempo, obj
            )
            alpha = np.clip((n_filled - 4.0) / 6.0, 0.0, 1.0)
            mean_w = ((1.0 - alpha) * mean_w.astype(np.float64) + alpha * meta_z).astype(np.float32)
            std_w = np.sqrt(std_w.astype(np.float64) ** 2 + (alpha * fam_std) ** 2).astype(np.float32)

        # Rank-adaptive duration reweighting + side-bias correction for ranks outside trained Elo brackets
        need_elo_adj = any(ELO_TO_IDX.get(e, 3) not in self.trained_elo_indices for e in raw_elos)
        if need_elo_adj and dur is not None and dur.shape[-1] == 4:
            p_dur = _np_sigmoid(dur.astype(np.float64))
            z_high = _np_logit(p_dur @ _HIGH_ELO_DUR_W)
            w_mat = np.array([_ELO_DUR_WEIGHTS.get(e, _HIGH_ELO_DUR_W) for e in raw_elos], dtype=np.float64)
            z_tgt = _np_logit((p_dur * w_mat).sum(axis=1))
            fill_scale = np.clip(n_filled / 6.0, 0.35, 1.0)
            side_off = np.array(
                [
                    _ELO_BLUE_SIDE_OFFSET.get(e, 0.0) if ELO_TO_IDX.get(e, 3) not in self.trained_elo_indices else 0.0
                    for e in raw_elos
                ],
                dtype=np.float64,
            )
            mean_w = (mean_w.astype(np.float64) + fill_scale * (z_tgt - z_high) + side_off).astype(np.float32)

        return {"mean_win_logit": mean_w, "std_win_logit": std_w, "duration_logits": dur}