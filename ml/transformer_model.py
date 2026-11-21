import torch
import torch.nn as nn
import torch.nn.functional as F

MECH_DIM = 10  # [range, phys, mag, true, mit, cc, heal_shield, mobility, waveclear, scale]


class RelBiasTransformerBlock(nn.Module):
    """Pre-LN Transformer block with exact Blue/Red-equivariant (ally/enemy x role_i x role_j) attention bias."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.22):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.rel_bias = nn.Embedding(50, nhead)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rel_idx: torch.Tensor, key_mask_bias: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(-1, 10, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        bias = self.rel_bias(rel_idx).permute(2, 0, 1).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias + key_mask_bias
        attn = self.attn_drop(torch.softmax(scores, dim=-1))

        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(-1, 10, self.d_model)
        x = x + self.proj_drop(self.proj(ctx))
        x = x + self.ffn(self.norm2(x))
        return x


class DraftSetTransformer(nn.Module):
    """
    v4 Two-Stage Pretrainable Patch-Delta Set Transformer:
      - Decomposed Order-1 Strength: global rc_bias + patch-specific rc_patch_delta (latest patch folded at ONNX export)
      - Compact Shared-Backbone Order-2 Counter & Synergy Projections (pretrained on 100%-populated GD@15/XPD@15)
      - Lean d_model=96 Equivariant RelBias Transformer + Duration-Marginalized Win Head
    """
    def __init__(
        self,
        num_champs: int = 250,
        num_patches: int = 12,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        fm_dim: int = 24,
        dropout: float = 0.22,
    ):
        super().__init__()
        self.num_champs = num_champs
        self.num_patches = max(1, num_patches)
        self.num_rc = 5 * num_champs + 1
        self.fm_scale = fm_dim ** -0.5

        # Order-1: Decomposed global + per-patch (role, champ) strength
        self.rc_bias = nn.Embedding(self.num_rc, 1, padding_idx=0)
        nn.init.zeros_(self.rc_bias.weight)
        self.rc_patch_delta = nn.Parameter(torch.zeros(self.num_patches, self.num_rc))

        # Shared champion + role + mechanics representation (pretrained in Stage 1 on dense physics)
        self.rc_emb = nn.Embedding(self.num_rc, d_model, padding_idx=0)
        self.champ_emb = nn.Embedding(num_champs, d_model, padding_idx=0)
        nn.init.normal_(self.rc_emb.weight, std=0.03)
        nn.init.normal_(self.champ_emb.weight, std=0.05)
        with torch.no_grad():
            self.rc_emb.weight[0].zero_()
            self.champ_emb.weight[0].zero_()

        self.mech_proj = nn.Sequential(
            nn.Linear(MECH_DIM, d_model), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(d_model, d_model)
        )
        self.role_emb = nn.Embedding(5, d_model)
        self.elo_emb = nn.Embedding(6, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Compact Order-2 Counter & Synergy projections off shared embeddings (no sparse param bloat)
        self.ctr_atk = nn.Linear(d_model, fm_dim)
        self.ctr_def = nn.Linear(d_model, fm_dim)
        self.syn_proj = nn.Linear(d_model, fm_dim)
        self.fm_drop = nn.Dropout(dropout)

        ctr_init = torch.full((5, 5), 0.12)
        ctr_init.fill_diagonal_(0.55)
        ctr_init[3, 4] = ctr_init[4, 3] = 0.35
        ctr_init[1, 2] = ctr_init[2, 1] = 0.25
        self.role_ctr_w = nn.Parameter(ctr_init)
        self.role_syn_w = nn.Parameter(torch.full((5, 5), 0.20))
        self.fm_lane_tempo = nn.Parameter(torch.tensor([0.40, 0.35], dtype=torch.float32))
        self.register_buffer("off_diag", 1.0 - torch.eye(5))

        rel = torch.zeros((10, 10), dtype=torch.long)
        for i in range(10):
            for j in range(10):
                same = 0 if ((i < 5) == (j < 5)) else 1
                rel[i, j] = same * 25 + (i % 5) * 5 + (j % 5)
        self.register_buffer("rel_idx", rel)

        self.blocks = nn.ModuleList([RelBiasTransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pool_gate = nn.Linear(d_model, 1)

        # Antisymmetric Win & Duration heads (fine-tuned in Stage 2)
        self.team_win_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        self.lane_win_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
        self.elo_side_bias = nn.Embedding(6, 1)
        nn.init.zeros_(self.elo_side_bias.weight)

        self.dur_delta_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 4)
        )
        self.dur_side_bias = nn.Parameter(torch.zeros(4))
        self.dur_cls_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 4)
        )
        self.marg_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        # Dense Stage-1 Physics heads
        self.lane_tempo_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )
        self.obj_head = nn.Sequential(nn.Linear(d_model * 3, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 4))
        self.obj_side_bias = nn.Parameter(torch.zeros(4))
        self.pstat_head = nn.Linear(d_model, 5)

    def reg_loss(self) -> torch.Tensor:
        """Empirical-Bayes L2 shrinkage: tighter on patch deltas so small-N shifts shrink cleanly to global."""
        return (
            1.4e-3 * (self.rc_bias.weight ** 2).sum()
            + 4.5e-3 * (self.rc_patch_delta ** 2).sum()
            + 2.0e-4 * (self.rc_emb.weight ** 2).sum()
        )

    def forward(self, champ_idx, mech_vec, role_idx, team_idx, setup_idx, elo_idx, l1_prior, patch_idx=None):
        valid_bool = champ_idx > 0
        valid = valid_bool.float()
        rc_idx = torch.where(valid_bool, role_idx * self.num_champs + champ_idx, torch.zeros_like(champ_idx))

        if self.training:
            keep_id = (torch.rand_like(valid) >= 0.12).float()
        else:
            keep_id = torch.ones_like(valid)
        id_mask = (valid * keep_id).unsqueeze(-1)
        v_mask = valid.unsqueeze(-1)

        # --- Order 1: Global + Patch-Specific Strength (patch 0 = latest meta at ONNX inference) ---
        if patch_idx is not None:
            p_delta = self.rc_patch_delta[patch_idx.unsqueeze(1), rc_idx]
        else:
            p_delta = F.embedding(rc_idx, self.rc_patch_delta[0].unsqueeze(-1)).squeeze(-1)
        rc_w = (self.rc_bias(rc_idx).squeeze(-1) + p_delta) * valid * keep_id
        wide_logit = rc_w[:, :5].sum(dim=1) - rc_w[:, 5:].sum(dim=1)

        # --- Shared Token Representations ---
        tok_id = (self.rc_emb(rc_idx) + self.champ_emb(champ_idx)) * id_mask
        tok_base = tok_id + self.mech_proj(mech_vec) * v_mask + self.role_emb(role_idx) * v_mask

        # --- Order 2: Compact Shared-Backbone Counter & Synergy FM ---
        atk = self.fm_drop(self.ctr_atk(tok_base)) * v_mask
        dfn = self.fm_drop(self.ctr_def(tok_base)) * v_mask
        m_br = torch.bmm(atk[:, :5, :], dfn[:, 5:, :].transpose(1, 2))
        m_rb = torch.bmm(atk[:, 5:, :], dfn[:, :5, :].transpose(1, 2))
        ctr_mat = (m_br - m_rb.transpose(1, 2)) * self.fm_scale
        w_ctr = 0.5 * (self.role_ctr_w + self.role_ctr_w.transpose(0, 1))
        ctr_logit = (ctr_mat * w_ctr.unsqueeze(0)).sum(dim=(1, 2))
        lane_ctr_diag = torch.diagonal(ctr_mat, dim1=1, dim2=2)

        syn = self.fm_drop(self.syn_proj(tok_base)) * v_mask
        sb = torch.bmm(syn[:, :5, :], syn[:, :5, :].transpose(1, 2)) * self.fm_scale
        sr = torch.bmm(syn[:, 5:, :], syn[:, 5:, :].transpose(1, 2)) * self.fm_scale
        w_syn = 0.5 * (self.role_syn_w + self.role_syn_w.transpose(0, 1)) * self.off_diag
        syn_logit = 0.5 * ((sb - sr) * w_syn.unsqueeze(0)).sum(dim=(1, 2))

        # --- Order 3+: Equivariant Set Transformer ---
        x = self.emb_drop(tok_base)
        key_mask_bias = ((1.0 - valid) * -1e4).unsqueeze(1).unsqueeze(2)
        for blk in self.blocks:
            x = blk(x, self.rel_idx, key_mask_bias)
        h = self.norm(x) * v_mask

        blue_h, red_h = h[:, :5, :], h[:, 5:, :]
        wb = torch.softmax(self.pool_gate(blue_h) + ((1.0 - valid[:, :5]) * -1e4).unsqueeze(-1), dim=1)
        wr = torch.softmax(self.pool_gate(red_h) + ((1.0 - valid[:, 5:]) * -1e4).unsqueeze(-1), dim=1)
        tb = (blue_h * wb).sum(dim=1)
        tr = (red_h * wr).sum(dim=1)

        elo_ctx = self.elo_emb(elo_idx)
        team_b = torch.cat([tb, tb - tr, tb * tr], dim=-1)
        team_r = torch.cat([tr, tr - tb, tr * tb], dim=-1)
        lane_b = torch.cat([blue_h, blue_h - red_h, blue_h * red_h], dim=-1)
        lane_r = torch.cat([red_h, red_h - blue_h, red_h * blue_h], dim=-1)

        team_logit = (
            self.team_win_head(torch.cat([team_b, elo_ctx], dim=-1))
            - self.team_win_head(torch.cat([team_r, elo_ctx], dim=-1))
        ).squeeze(-1)
        lane_mask = valid[:, :5] * valid[:, 5:]
        lane_logit = ((self.lane_win_head(lane_b) - self.lane_win_head(lane_r)).squeeze(-1) * lane_mask).sum(dim=1)
        side_b = self.elo_side_bias(elo_idx).squeeze(-1)

        direct_logit = wide_logit + ctr_logit + syn_logit + team_logit + lane_logit + side_b

        dur_delta = self.dur_delta_head(team_b) - self.dur_delta_head(team_r)
        dur_win = direct_logit.unsqueeze(-1) + dur_delta + self.dur_side_bias
        dur_cls = self.dur_cls_head(torch.cat([tb + tr, torch.abs(tb - tr), tb * tr], dim=-1))
        dur_probs = torch.softmax(dur_cls, dim=-1)
        p_marg = (dur_probs * torch.sigmoid(dur_win)).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
        z_marg = torch.log(p_marg / (1.0 - p_marg))

        g = torch.sigmoid(self.marg_gate) * 0.5
        win_logit = (1.0 - g) * direct_logit + g * z_marg

        tempo_diff = (self.lane_tempo_head(lane_b) - self.lane_tempo_head(lane_r)) + lane_ctr_diag.unsqueeze(-1) * self.fm_lane_tempo
        obj_pred = (self.obj_head(team_b) - self.obj_head(team_r)) + self.obj_side_bias
        pstat_pred = self.pstat_head(h)

        return win_logit, dur_win, tempo_diff, dur_cls, obj_pred, pstat_pred


class DraftEnsembleONNX(nn.Module):
    """Wraps 5 calibrated v4 Two-Stage Patch-Delta Transformers into one 6-output ONNX graph."""
    def __init__(self, models: list[DraftSetTransformer], temps: list[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        t_vals = temps if temps and len(temps) == len(models) else [1.0] * len(models)
        self.register_buffer("inv_temps", torch.tensor([1.0 / max(0.25, t) for t in t_vals], dtype=torch.float32).view(-1, 1))
        self.register_buffer("zero_keep", torch.zeros(1, dtype=torch.float32))

    def forward(self, champ_idx, mech_vec, role_idx, team_idx, setup_idx, elo_idx, l1_prior):
        w_list, d_list, dp_list, tm_list, ob_list = [], [], [], [], []
        for m in self.models:
            w, dur, tm, dc, ob, _ = m(champ_idx, mech_vec, role_idx, team_idx, setup_idx, elo_idx, l1_prior)
            w_list.append(w)
            d_list.append(dur)
            dp_list.append(torch.softmax(dc, dim=-1))
            tm_list.append(tm)
            ob_list.append(ob)
        w_stack = torch.stack(w_list, dim=0) * self.inv_temps
        keep = self.zero_keep * (l1_prior + team_idx[:, 0].float() + setup_idx[:, 0].float())
        return (
            w_stack.mean(dim=0) + keep,
            w_stack.std(dim=0, unbiased=False),
            torch.stack(d_list, dim=0).mean(dim=0),
            torch.stack(dp_list, dim=0).mean(dim=0),
            torch.stack(tm_list, dim=0).mean(dim=0),
            torch.stack(ob_list, dim=0).mean(dim=0),
        )