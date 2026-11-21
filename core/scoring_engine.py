import difflib, itertools, math, re
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

ROLES = ["TOP", "JGL", "MID", "BOT", "SUP"]

# Exact relative spreads from the Matchups & Synergies heatmaps (m = 1 + pct)
MATCHUP_HEATMAP: Dict[str, Dict[str, float]] = {
    "TOP": {"TOP": 2.09, "JGL": 0.85, "MID": 0.82, "BOT": 0.67, "SUP": 0.58},
    "JGL": {"TOP": 0.88, "JGL": 1.28, "MID": 1.08, "BOT": 0.78, "SUP": 0.98},
    "MID": {"TOP": 0.78, "JGL": 0.99, "MID": 1.71, "BOT": 0.73, "SUP": 0.79},
    "BOT": {"TOP": 0.73, "JGL": 0.85, "MID": 0.85, "BOT": 1.29, "SUP": 1.28},
    "SUP": {"TOP": 0.61, "JGL": 0.99, "MID": 0.86, "BOT": 1.19, "SUP": 1.36},
}

SYNERGY_HEATMAP: Dict[str, Dict[str, float]] = {
    "TOP": {"JGL": 1.12, "MID": 1.05, "BOT": 0.85, "SUP": 0.97},
    "JGL": {"TOP": 0.87, "MID": 1.24, "BOT": 0.94, "SUP": 0.95},
    "MID": {"TOP": 0.76, "JGL": 1.28, "BOT": 0.96, "SUP": 1.00},
    "BOT": {"TOP": 0.61, "JGL": 0.83, "MID": 0.83, "SUP": 1.73},
    "SUP": {"TOP": 0.71, "JGL": 0.95, "MID": 0.82, "BOT": 1.53},
}

SPELL_NAMES = {
    1: "Cleanse", 3: "Exhaust", 4: "Flash", 6: "Ghost", 7: "Heal",
    11: "Smite", 12: "TP", 14: "Ignite", 21: "Barrier",
}

# Controls low-Elo execution floor, scaling, teamfight emphasis, and Expectimax pool dampening (0.0 on Diamond/Master).
LOW_ELO_SCALE: Dict[str, float] = {
    "IRON": 1.00, "BRONZE": 0.90, "SILVER": 0.75, "GOLD": 0.55,
    "PLATINUM": 0.30, "EMERALD": 0.12, "DIAMOND": 0.0, "MASTER": 0.0,
}


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    z = max(-15.0, min(15.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class SetupOption:
    keystone_id: int
    keystone_name: str
    substyle_id: int
    substyle_name: str
    spell_pair: str
    delta_logit: float
    games: int


@dataclass
class ChampProfile:
    champ_id: int
    name: str
    role: str
    base_wr: float
    play_rate: float
    comp_vec: np.ndarray
    mech_vec: np.ndarray = field(default_factory=lambda: np.full(10, 0.4))
    base_games: int = 500
    duration_wr: np.ndarray = field(default_factory=lambda: np.full(4, 0.5))
    blue_side_delta: float = 0.0
    elo_wr_delta: Dict[str, float] = field(default_factory=dict)
    elo_play_rate: Dict[str, float] = field(default_factory=dict)
    mechanical_floor: float = 0.5
    otp_bias_logit: float = 0.0
    mastery_slope: float = 0.03
    baseline_log_m: float = 4.0
    synergy_spread: Dict[str, float] = field(default_factory=dict)
    matchup_spread: float = 0.06


@dataclass
class PersonalProfile:
    mastery_points: Dict[int, int] = field(default_factory=dict)
    mastery_logit: Dict[int, float] = field(default_factory=dict)
    recent_games_45d: Dict[int, int] = field(default_factory=dict)
    pool_ids: Set[int] = field(default_factory=set)
    duration_slope: float = 0.0
    personal_matchups: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)


@dataclass
class DraftState:
    my_role: str
    my_pick_index: int
    is_blue_side: bool = True
    phase: str = "PICK"                         # "BAN" | "PICK"
    elo_tier: str = "EMERALD"
    role_status: str = "PRIMARY"                # "PRIMARY" | "SECONDARY" | "AUTOFILL"
    locked_allies: Dict[str, int] = field(default_factory=dict)
    hovered_allies: Dict[str, int] = field(default_factory=dict)
    enemy_locks: List[int] = field(default_factory=list)
    ally_bans: Set[int] = field(default_factory=set)
    enemy_bans: Set[int] = field(default_factory=set)
    mastery_mode: str = "POOL"                  # "OFF" | "POOL" | "ON"
    mastery_weight: float = 0.75
    lcb_kappa: float = 0.75
    hover_confidence: float = 0.75
    n_enemy_pool: int = 7
    expectimax_depth: int = 1
    xp_stall_mode: bool = False
    autofill_protection: bool = True
    focus_names: List[str] = field(default_factory=list)


class DraftScoringEngine:
    def __init__(
        self,
        champions: Dict[Tuple[str, int], ChampProfile],
        precomputed_matchups: Dict[Tuple[str, int, str, int], float],
        precomputed_synergies: Dict[Tuple[str, int, str, int], float],
        matchup_kish: Optional[Dict[Tuple[str, int, str, int], float]] = None,
        synergy_kish: Optional[Dict[Tuple[str, int, str, int], float]] = None,
        ban_likelihoods: Optional[Dict[Tuple[int, int, str, int], float]] = None,
        setup_table: Optional[Dict[Tuple[str, int, int], List[SetupOption]]] = None,
        personal: Optional[PersonalProfile] = None,
        onnx_net=None,
    ):
        self.champions = champions
        self.precomputed_matchups = precomputed_matchups
        self.precomputed_synergies = precomputed_synergies
        self.matchup_kish = matchup_kish or {}
        self.synergy_kish = synergy_kish or {}
        self.ban_likelihoods = ban_likelihoods or {}
        self.setup_table = setup_table or {}
        self.personal = personal or PersonalProfile()
        self.onnx_net = onnx_net

        self.by_role: Dict[str, List[ChampProfile]] = {r: [] for r in ROLES}
        self.roles_by_cid: Dict[int, Dict[str, float]] = {}
        self.norm_name_to_cid: Dict[str, Tuple[int, str]] = {}
        self.cid_best_profile: Dict[int, ChampProfile] = {}

        for (role, cid), prof in champions.items():
            self.by_role[role].append(prof)
            self.roles_by_cid.setdefault(cid, {})[role] = prof.play_rate
            self.norm_name_to_cid[_norm_name(prof.name)] = (cid, prof.name)
            prev = self.cid_best_profile.get(cid)
            if prev is None or prof.play_rate > prev.play_rate:
                self.cid_best_profile[cid] = prof

    @staticmethod
    def _low_elo_scale(elo_tier: str) -> float:
        return LOW_ELO_SCALE.get(str(elo_tier).upper(), 0.0)

    @staticmethod
    def _eff_play_rate(prof: ChampProfile, elo_tier: str) -> float:
        return prof.elo_play_rate.get(str(elo_tier).upper(), prof.play_rate)

    # ------------------------------------------------------------------
    # Fuzzy Champion Shortlist Resolver ("lilia" -> "Lillia", etc.)
    # ------------------------------------------------------------------
    def resolve_focus_champions(self, focus_names: List[str]) -> Tuple[List[Tuple[int, str]], List[str]]:
        resolved: List[Tuple[int, str]] = []
        seen_cids: Set[int] = set()
        unmatched: List[str] = []
        all_norms = list(self.norm_name_to_cid.keys())

        for raw in focus_names:
            q = _norm_name(raw)
            if not q:
                continue
            match_key = None
            if q in self.norm_name_to_cid:
                match_key = q
            else:
                if len(q) >= 3:
                    prefixes = [k for k in all_norms if k.startswith(q)]
                    if prefixes:
                        match_key = min(prefixes, key=len)
                if match_key is None:
                    close = difflib.get_close_matches(q, all_norms, n=1, cutoff=0.72)
                    if close:
                        match_key = close[0]

            if match_key is not None:
                cid, disp_name = self.norm_name_to_cid[match_key]
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    resolved.append((cid, disp_name))
            else:
                unmatched.append(raw.strip())

        return resolved, unmatched

    # ------------------------------------------------------------------
    # Bayesian Ban-to-Pick Inference (Enemy & Ally)
    # ------------------------------------------------------------------
    def infer_hidden_role_posterior(
        self, role: str, bans: Set[int], unavailable: Set[int], is_same_team: bool = True, elo_tier: str = "EMERALD"
    ) -> List[Tuple[ChampProfile, float]]:
        avail = [c for c in self.by_role[role] if c.champ_id not in unavailable]
        if not avail:
            return []
        same_flag = 1 if is_same_team else 0
        weights = []
        for c in avail:
            log_w = math.log(max(self._eff_play_rate(c, elo_tier), 1e-4))
            for b in bans:
                log_w += math.log(max(self.ban_likelihoods.get((same_flag, b, role, c.champ_id), 0.015), 1e-3))
            weights.append(log_w)
        max_lw = max(weights)
        probs = [math.exp(w - max_lw) for w in weights]
        z = sum(probs) + 1e-12
        return [(c, p / z) for c, p in zip(avail, probs)]

    def top_ban_inference_alerts(self, state: DraftState) -> List[str]:
        if not state.enemy_bans:
            return []
        unavail = state.ally_bans | state.enemy_bans | set(state.locked_allies.values()) | set(state.enemy_locks)
        alerts = []
        for role in ROLES:
            post = self.infer_hidden_role_posterior(role, state.enemy_bans, unavail, is_same_team=True, elo_tier=state.elo_tier)
            base_sum = sum(self._eff_play_rate(c, state.elo_tier) for c, _ in post) + 1e-9
            for c, p_post in post:
                base_p = self._eff_play_rate(c, state.elo_tier) / base_sum
                if p_post >= 0.14 and p_post / max(base_p, 1e-4) >= 1.85:
                    alerts.append((p_post, f"{c.name} ({role} {int(p_post*100)}%)"))
        alerts.sort(reverse=True)
        return [txt for _, txt in alerts[:3]]

    # ------------------------------------------------------------------
    # Soft Hover Priors (championPickIntent -> 75/25 probability mix)
    # ------------------------------------------------------------------
    def get_ally_soft_distribution(
        self, role: str, state: DraftState, unavailable: Set[int]
    ) -> List[Tuple[ChampProfile, float]]:
        h_cid = state.hovered_allies.get(role, 0)
        base_post = self.infer_hidden_role_posterior(role, state.ally_bans, unavailable, is_same_team=True, elo_tier=state.elo_tier)
        if h_cid <= 0 or h_cid in unavailable or (role, h_cid) not in self.champions:
            return base_post[:10]

        h_prof = self.champions[(role, h_cid)]
        p_h = state.hover_confidence
        others = [(c, p) for c, p in base_post if c.champ_id != h_cid]
        other_sum = sum(p for _, p in others) + 1e-12
        return [(h_prof, p_h)] + [(c, (1.0 - p_h) * (p / other_sum)) for c, p in others[:8]]

    # ------------------------------------------------------------------
    # Clean 120-Permutation Flex-Pick Solver
    # ------------------------------------------------------------------
    def solve_enemy_permutations(self, state: DraftState, min_prob: float = 0.01) -> List[Tuple[Dict[str, int], float]]:
        locks = [cid for cid in state.enemy_locks if cid > 0]
        if not locks:
            return [({}, 1.0)]

        scored = []
        for perm in itertools.permutations(ROLES, len(locks)):
            log_p = 0.0
            mapping = {}
            for cid, role in zip(locks, perm):
                mapping[role] = cid
                prof = self.champions.get((role, cid))
                pr = self._eff_play_rate(prof, state.elo_tier) if prof else 1e-4
                log_p += math.log(max(pr, 1e-4))
                for b in state.enemy_bans:
                    log_p += 0.55 * math.log(max(self.ban_likelihoods.get((1, b, role, cid), 0.015), 1e-3))
            scored.append((mapping, log_p))

        max_lp = max(lp for _, lp in scored)
        exps = [(m, math.exp(lp - max_lp)) for m, lp in scored]
        total = sum(p for _, p in exps) + 1e-12
        normed = [(m, p / total) for m, p in exps]
        pruned = sorted([(m, p) for m, p in normed if p >= min_prob], key=lambda x: x[1], reverse=True)
        if not pruned:
            return [normed[0]]
        renorm = sum(p for _, p in pruned)
        return [(m, p / renorm) for m, p in pruned[:8]]

    # ------------------------------------------------------------------
    # Optimal-Setup Evaluation + Patch-Zero Mechanical Similarity Fallback
    # ------------------------------------------------------------------
    def best_setup(self, role: str, cid: int, lane_enemy_cid: Optional[int], enemy_map: Dict[str, int]) -> Tuple[float, str, List[str]]:
        options = [o for o in self.setup_table.get((role, cid, lane_enemy_cid), []) if o.games >= 15] if lane_enemy_cid else []
        if not options:
            options = self.setup_table.get((role, cid, 0), [])
        if options:
            best = max(options, key=lambda o: o.delta_logit)
            s1, s2 = [SPELL_NAMES.get(int(x), x) for x in best.spell_pair.split("-")]
            setup_str = f"{best.keystone_name} + {best.substyle_name} ({s1}/{s2})"
            setup_z = max(-0.04, min(0.08, best.delta_logit))
        else:
            setup_str, setup_z = "Standard Setup", 0.0

        enemy_vecs = [self.champions[(er, ec)].comp_vec for er, ec in enemy_map.items() if (er, ec) in self.champions]
        item_flags = []
        if enemy_vecs:
            e_sum = sum(enemy_vecs, np.zeros(6))
            if e_sum[2] >= 1.6:
                item_flags.append("%HP Shred")
            if len(enemy_vecs) >= 3 and e_sum[1] / max(0.1, e_sum[0] + e_sum[1]) > 0.65:
                item_flags.append("Early MR")
            elif len(enemy_vecs) >= 3 and e_sum[0] / max(0.1, e_sum[0] + e_sum[1]) > 0.72:
                item_flags.append("Early Armor")
        return setup_z, setup_str, item_flags

    def _delta_and_var(
        self, my_role: str, cid: int, other_role: str, other_cid: int, is_enemy: bool,
        cand_override: Optional[ChampProfile] = None
    ) -> Tuple[float, float]:
        key = (my_role, cid, other_role, other_cid)
        table = self.precomputed_matchups if is_enemy else self.precomputed_synergies
        k_tbl = self.matchup_kish if is_enemy else self.synergy_kish

        if key in table:
            d = table[key]
            n_kish = k_tbl.get(key, 40.0)
        else:
            cand = cand_override or self.champions.get((my_role, cid))
            d, n_kish = 0.0, 5.0
            if cand is not None:
                sims = []
                c_norm = float(np.linalg.norm(cand.mech_vec)) + 1e-9
                for peer in self.by_role[my_role]:
                    pk = (my_role, peer.champ_id, other_role, other_cid)
                    if peer.champ_id != cid and pk in table:
                        sim = float(np.dot(cand.mech_vec, peer.mech_vec)) / (c_norm * (float(np.linalg.norm(peer.mech_vec)) + 1e-9))
                        if sim > 0.75:
                            sims.append((sim, table[pk]))
                if sims:
                    sims.sort(reverse=True)
                    top3 = sims[:3]
                    d = sum(s * val for s, val in top3) / sum(s for s, _ in top3)
                    n_kish = 18.0

        if is_enemy and my_role == other_role:
            rec = self.personal.personal_matchups.get((cid, other_cid))
            if rec and rec[1] >= 5:
                w_p, n_p = rec
                alpha = n_p / (n_p + 25.0)
                d = (1.0 - alpha) * d + alpha * logit((w_p + 1.0) / (n_p + 2.0))
                n_kish += n_p

        hm = MATCHUP_HEATMAP[my_role][other_role] if is_enemy else SYNERGY_HEATMAP[my_role][other_role]
        k_prior = 450.0 / (hm ** 2)
        return d, 4.0 / (n_kish + k_prior)

    def _comp_score_and_tail_var(
        self, cand: ChampProfile, allies_locked: Dict[str, int], ally_hover_vecs: List[np.ndarray],
        enemies: Dict[str, int], state: DraftState
    ) -> Tuple[float, float, List[str]]:
        ally_vecs = [self.champions[(r, c)].comp_vec for r, c in allies_locked.items() if (r, c) in self.champions]
        enemy_vecs = [self.champions[(r, c)].comp_vec for r, c in enemies.items() if (r, c) in self.champions]
        team = sum(ally_vecs, np.zeros(6)) + sum(ally_hover_vecs, np.zeros(6)) + cand.comp_vec
        e_team = sum(enemy_vecs, np.zeros(6))
        n_eff_allies = len(ally_vecs) + len(ally_hover_vecs) + 1
        flags, score, tail_var = [], 0.0, 0.0

        low_s = self._low_elo_scale(state.elo_tier)
        # Front-to-back teamfight deficits punish lower-Elo 5v5s harder than high-Elo sidelane macro
        comp_mult = 1.0 + 0.35 * low_s

        tot_d = team[0] + team[1] + 1e-6
        phys_s, mag_s = team[0] / tot_d, team[1] / tot_d
        e_tank = min(2.5, e_team[2])

        if n_eff_allies >= 3 and (phys_s > 0.78 or mag_s > 0.78):
            excess = max(phys_s, mag_s) - 0.78
            score -= comp_mult * (0.14 + 0.85 * excess) * (0.55 + e_tank)
            tail_var += 0.0025 * (1.0 + e_tank)
            flags.append(("Full AD" if phys_s > mag_s else "Full AP") + (" vs Tanks" if e_tank > 0.85 else ""))

        if n_eff_allies >= 4 and team[3] < 1.15:
            score -= comp_mult * (0.09 + 0.05 * max(0.0, 1.15 - team[3]))
            tail_var += 0.0012
            flags.append("Low Hard CC")
        if n_eff_allies >= 4 and team[2] < 0.75:
            score -= comp_mult * (0.07 + 0.05 * max(0.0, 0.75 - team[2]))
            tail_var += 0.0012
            flags.append("No Frontline")

        late_edge = float((cand.duration_wr[2] + cand.duration_wr[3]) - (cand.duration_wr[0] + cand.duration_wr[1]))
        if state.xp_stall_mode:
            score += 0.06 * cand.comp_vec[4] + 0.08 * cand.comp_vec[5] + 0.40 * late_edge
        else:
            if low_s > 0.0:
                score += 0.14 * low_s * late_edge + 0.025 * low_s * cand.comp_vec[5]
            if state.mastery_mode == "ON" and abs(self.personal.duration_slope) > 0.02:
                score += 0.60 * self.personal.duration_slope * late_edge

        return score, tail_var, flags

    def _flex_masking_bonus(self, cand: ChampProfile, open_enemies: List[str]) -> float:
        if not open_enemies:
            return 0.0
        role_prs = self.roles_by_cid.get(cand.champ_id, {})
        tot = sum(role_prs.values()) + 1e-9
        probs = [v / tot for v in role_prs.values() if v / tot > 0.08]
        if len(probs) <= 1:
            return 0.0
        return 0.032 * (-sum(p * math.log(p) for p in probs)) * (len(open_enemies) / 5.0)

    def _pinch_denial_bonus(self, cand: ChampProfile, enemy_map: Dict[str, int], open_enemies: List[str], elo_tier: str) -> float:
        if len(enemy_map) < 2 or not open_enemies:
            return 0.0
        enemy_vecs = [self.champions[(er, ec)].comp_vec for er, ec in enemy_map.items() if (er, ec) in self.champions]
        e_sum = sum(enemy_vecs, np.zeros(6))
        e_open_pr = sum(
            self._eff_play_rate(self.champions[(er, cand.champ_id)], elo_tier)
            for er in open_enemies if (er, cand.champ_id) in self.champions
        )
        if e_open_pr < 0.03:
            return 0.0
        pinch = 0.0
        if e_sum[1] / (e_sum[0] + e_sum[1] + 1e-6) < 0.22 and cand.comp_vec[1] > 0.7:
            pinch += 0.035
        if e_sum[2] < 0.5 and cand.comp_vec[2] > 0.75:
            pinch += 0.030
        return pinch * min(1.5, e_open_pr * 12.0)

    def _expectimax_hidden_risk(
        self, cand: ChampProfile, state: DraftState, enemy_map: Dict[str, int], unavailable: Set[int]
    ) -> Tuple[float, float, float]:
        open_enemies = [r for r in ROLES if r not in enemy_map]
        open_allies = [r for r in ROLES if r not in state.locked_allies and r != state.my_role]

        # Lower-Elo opponents hold narrower pools and counter-pick less reliably than Master+ players
        low_s = self._low_elo_scale(state.elo_tier)
        eff_pool = max(2.0, float(state.n_enemy_pool) * (1.0 - 0.50 * low_s))

        enemy_risk_z, lane_counter_prob = 0.0, 0.0
        for er in open_enemies:
            posterior = self.infer_hidden_role_posterior(er, state.enemy_bans, unavailable, is_same_team=True, elo_tier=state.elo_tier)
            if not posterior:
                continue
            entries, counter_mass = [], 0.0
            for enemy_c, post_w in posterior:
                d, _ = self._delta_and_var(state.my_role, cand.champ_id, er, enemy_c.champ_id, True, cand)
                entries.append((d, post_w, enemy_c))
                if d < -0.08:
                    counter_mass += post_w

            entries.sort(key=lambda x: x[0])
            cum_f, exp_worst = 0.0, 0.0
            for d, w, _ in entries:
                next_f = min(1.0, cum_f + w)
                exp_worst += (((1.0 - cum_f) ** eff_pool) - ((1.0 - next_f) ** eff_pool)) * d
                cum_f = next_f

            if state.expectimax_depth >= 2 and er == state.my_role and state.locked_allies:
                ply2 = sum(
                    w * min(0.0, sum(self._delta_and_var(ar, ac, er, ec.champ_id, True)[0] for ar, ac in state.locked_allies.items()))
                    for _, w, ec in entries[:3]
                )
                exp_worst += 0.25 * ply2

            enemy_risk_z += (MATCHUP_HEATMAP[er][state.my_role] / 5.0) * min(0.0, exp_worst)
            if er == state.my_role:
                lane_counter_prob = 1.0 - ((1.0 - counter_mass) ** eff_pool)

        ally_risk_z, hover_syn_z = 0.0, 0.0
        for ar in open_allies:
            ally_w = SYNERGY_HEATMAP[state.my_role][ar] / 4.0
            h_cid = state.hovered_allies.get(ar, 0)
            has_valid_hover = h_cid > 0 and h_cid not in unavailable and (ar, h_cid) in self.champions
            a_dist = self.get_ally_soft_distribution(ar, state, unavailable)

            if has_valid_hover:
                exp_syn = sum(p * self._delta_and_var(state.my_role, cand.champ_id, ar, ac.champ_id, False, cand)[0] for ac, p in a_dist)
                hover_syn_z += 0.85 * exp_syn
                ally_risk_z -= 0.35 * (1.0 - state.hover_confidence) * ally_w * cand.synergy_spread.get(ar, 0.03)
            else:
                if state.ally_bans and a_dist:
                    exp_syn = sum(p * self._delta_and_var(state.my_role, cand.champ_id, ar, ac.champ_id, False, cand)[0] for ac, p in a_dist)
                    ally_risk_z += 0.35 * ally_w * exp_syn
                ally_risk_z -= 0.35 * ally_w * cand.synergy_spread.get(ar, 0.03)

        return enemy_risk_z + ally_risk_z, lane_counter_prob, hover_syn_z

    def _causal_mastery_adjustment(self, cand: ChampProfile, state: DraftState) -> Tuple[float, float]:
        otp_debias = -cand.otp_bias_logit
        pers_z = 0.0

        if state.autofill_protection and state.role_status in ("AUTOFILL", "SECONDARY"):
            scale = 1.0 if state.role_status == "AUTOFILL" else 0.45
            pers_z -= scale * (0.12 * cand.mechanical_floor + 0.65 * cand.matchup_spread)

        # Rank-adaptive mechanical execution penalty for lower Elos (bypassed if you actually main the champ)
        low_s = self._low_elo_scale(state.elo_tier)
        if low_s > 0.0 and cand.mechanical_floor > 0.50:
            my_pts = self.personal.mastery_points.get(cand.champ_id, 0)
            recent = self.personal.recent_games_45d.get(cand.champ_id, 0)
            if state.mastery_mode == "ON":
                mastery_credit = min(1.0, max(0.0, (math.log10(my_pts + 100.0) - 2.0) / 2.7) * 0.70 + min(recent, 15) / 15.0 * 0.50)
            elif state.mastery_mode == "POOL" and cand.champ_id in self.personal.pool_ids:
                mastery_credit = 0.55
            else:
                mastery_credit = 0.0
            pers_z -= low_s * (1.0 - mastery_credit) * 0.16 * (cand.mechanical_floor - 0.50)

        if state.mastery_mode == "ON":
            my_pts = self.personal.mastery_points.get(cand.champ_id, 0)
            my_log_m = math.log10(my_pts + 100.0)
            causal_skill = cand.mastery_slope * (my_log_m - cand.baseline_log_m)
            recent = self.personal.recent_games_45d.get(cand.champ_id, 0)
            recency_adj = 0.022 * math.log1p(min(recent, 40)) - 0.025
            wr_adj = 0.45 * self.personal.mastery_logit.get(cand.champ_id, 0.0)
            pers_z += state.mastery_weight * max(-0.16, min(0.18, causal_skill + recency_adj + wr_adj))

        return otp_debias, pers_z

    # ------------------------------------------------------------------
    # Unified Single-Pass Candidate Scorer (L1 + Stacker v2 / ONNX + LCB)
    # ------------------------------------------------------------------
    def _score_candidates(
        self, candidates: List[ChampProfile], state: DraftState,
        unavailable: Set[int], off_role_cids: Set[int]
    ) -> Dict[int, dict]:
        if not candidates:
            return {}
        perms = self.solve_enemy_permutations(state, min_prob=0.01)

        # Up to top 2 plausible enemy flex permutations for batched Stacker v2 / ONNX scoring
        nn_perms = [(m, p) for m, p in perms[:2] if p >= 0.15] or [perms[0]]
        nn_p_sum = sum(p for _, p in nn_perms)
        nn_perms = [(m, p / nn_p_sum) for m, p in nn_perms]

        ally_base_z = sum(
            logit(self.champions[(ar, acid)].base_wr)
            for ar, acid in state.locked_allies.items()
            if (ar, acid) in self.champions
        )
        enemy_base_z = sum(
            prob * sum(logit(self.champions[(er, ec)].base_wr) for er, ec in e_map.items() if (er, ec) in self.champions)
            for e_map, prob in perms
        )
        other_board_z = ally_base_z - enemy_base_z

        ally_hover_vecs = []
        for ar, h_cid in state.hovered_allies.items():
            if ar not in state.locked_allies and ar != state.my_role and h_cid > 0 and h_cid not in unavailable:
                dist = self.get_ally_soft_distribution(ar, state, unavailable)
                if dist:
                    ally_hover_vecs.append(sum((p * ac.comp_vec for ac, p in dist), np.zeros(6)))

        scored_list, onnx_payloads = [], []
        use_nn = bool(self.onnx_net and self.onnx_net.active)
        elo_key = str(state.elo_tier).upper()

        for cand in candidates:
            base_z = logit(cand.base_wr)
            otp_debias_z, pers_z = self._causal_mastery_adjustment(cand, state)
            elo_wr_z = cand.elo_wr_delta.get(elo_key, 0.0)
            side_z = cand.blue_side_delta * (0.5 if state.is_blue_side else -0.5)

            l1_var = 4.0 / (cand.base_games + 25.0)
            known_syn = 0.0
            for ar, acid in state.locked_allies.items():
                d, v = self._delta_and_var(state.my_role, cand.champ_id, ar, acid, False, cand)
                known_syn += 0.85 * d
                l1_var += 0.40 * v

            known_ctr, blind_z, hover_syn_z, comp_z, flex_z, pinch_z, setup_z, lane_ctr_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            best_flags, best_setup_str, best_item_flags = [], "Standard Setup", []

            for p_idx, (e_map, prob) in enumerate(perms):
                open_e = [r for r in ROLES if r not in e_map]
                ctr_p, ctr_v = 0.0, 0.0
                for er, ec in e_map.items():
                    d, v = self._delta_and_var(state.my_role, cand.champ_id, er, ec, True, cand)
                    ctr_p += 0.85 * d
                    ctr_v += 0.45 * v
                b_z, l_p, h_syn = self._expectimax_hidden_risk(cand, state, e_map, unavailable | {cand.champ_id})
                c_z, tail_v, c_flags = self._comp_score_and_tail_var(cand, state.locked_allies, ally_hover_vecs, e_map, state)
                f_z = self._flex_masking_bonus(cand, open_e)
                p_z = self._pinch_denial_bonus(cand, e_map, open_e, elo_key)
                s_z, s_str, i_flags = self.best_setup(state.my_role, cand.champ_id, e_map.get(state.my_role), e_map)

                known_ctr += prob * ctr_p
                blind_z += prob * b_z
                hover_syn_z += prob * h_syn
                comp_z += prob * c_z
                flex_z += prob * f_z
                pinch_z += prob * p_z
                setup_z += prob * s_z
                lane_ctr_p += prob * l_p
                l1_var += prob * (ctr_v + tail_v)
                if p_idx == 0:
                    best_flags, best_setup_str, best_item_flags = list(c_flags), s_str, i_flags

            if cand.champ_id in off_role_cids:
                best_flags.append("Off-Role Est.")

            # Separate current-board value (overlaps with Stacker v2) from decision/Elo/personal adjustments (undiluted)
            l1_board_z = other_board_z + base_z + side_z + known_ctr + known_syn + comp_z
            decision_z = elo_wr_z + otp_debias_z + hover_syn_z + blind_z + flex_z + pinch_z + setup_z + pers_z
            l1_z = l1_board_z + decision_z
            l1_std = math.sqrt(max(1e-5, l1_var))

            if use_nn:
                my_team = {**state.locked_allies, state.my_role: cand.champ_id}
                for e_map, _ in nn_perms:
                    onnx_payloads.append({
                        "blue": my_team if state.is_blue_side else e_map,
                        "red": e_map if state.is_blue_side else my_team,
                        "elo": elo_key,
                        "l1_prior": l1_board_z if state.is_blue_side else -l1_board_z,
                    })

            lcb_z = l1_z - state.lcb_kappa * l1_std
            scored_list.append({
                "champ_id": cand.champ_id,
                "champ": cand.name,
                "l1_z": l1_z,
                "l1_board_z": l1_board_z,
                "decision_z": decision_z,
                "l1_std": l1_std,
                "est_wr": round(sigmoid(l1_z) * 100, 2),
                "lcb_wr": round(sigmoid(lcb_z) * 100, 2),
                "sigma_wr": round(l1_std * 25.0, 2),
                "breakdown_wr": {
                    "base": round((cand.base_wr - 0.5) * 100, 2),
                    "otp_debias": round(otp_debias_z * 25, 2),
                    "matchups": round(known_ctr * 25, 2),
                    "synergy": round((known_syn + hover_syn_z) * 25, 2),
                    "blind_risk": round((blind_z + flex_z + pinch_z) * 25, 2),
                    "comp": round((comp_z + elo_wr_z + side_z) * 25, 2),
                    "setup": round(setup_z * 25, 2),
                    "mastery": round(pers_z * 25, 2),
                },
                "setup": best_setup_str,
                "item_flags": best_item_flags,
                "lane_counter_prob": round(lane_ctr_p * 100, 1),
                "flags": best_flags,
            })

        if onnx_payloads and use_nn:
            preds = self.onnx_net.predict_batch(onnx_payloads, self.champions)
            if preds is not None:
                n_locked = len(state.locked_allies) + 1 + len(state.enemy_locks)
                fill = min(1.0, max(0.0, (n_locked - 1) / 7.0))
                w_nn_max = 0.82 if getattr(self.onnx_net, "stacker_active", False) else 0.72
                w_nn = 0.35 + (w_nn_max - 0.35) * fill
                side_sgn = 1.0 if state.is_blue_side else -1.0
                k_p = len(nn_perms)

                for i, r in enumerate(scored_list):
                    base_idx = i * k_p
                    nn_z = sum(nn_perms[j][1] * float(preds["mean_win_logit"][base_idx + j]) * side_sgn for j in range(k_p))
                    nn_var = sum(nn_perms[j][1] * (float(preds["std_win_logit"][base_idx + j]) ** 2) for j in range(k_p))
                    if k_p > 1:
                        nn_var += sum(
                            nn_perms[j][1] * ((float(preds["mean_win_logit"][base_idx + j]) * side_sgn - nn_z) ** 2)
                            for j in range(k_p)
                        )
                    fused_board_z = w_nn * nn_z + (1.0 - w_nn) * r["l1_board_z"]
                    fused_z = fused_board_z + r["decision_z"]
                    fused_std = math.sqrt(w_nn * nn_var + (1.0 - w_nn) * (r["l1_std"] ** 2))
                    r["est_wr"] = round(sigmoid(fused_z) * 100, 2)
                    r["sigma_wr"] = round(fused_std * 25.0, 2)
                    r["lcb_wr"] = round(sigmoid(fused_z - state.lcb_kappa * fused_std) * 100, 2)

        return {r["champ_id"]: r for r in scored_list}

    def recommend(self, state: DraftState, top_k: int = 5) -> List[dict]:
        """Standard overall recommender (used by ml/evaluate.py and when no shortlist is active)."""
        return self.recommend_with_shortlist(state, top_overall=top_k)["overall"]

    def recommend_with_shortlist(self, state: DraftState, top_overall: int = 2) -> dict:
        """
        Scores both your pre-game Session Shortlist AND the best overall picks in one unified pass.
        Returns: {shortlist, overall, unavailable_focus, unmatched_focus}.
        """
        banned = state.ally_bans | state.enemy_bans
        picked = set(state.locked_allies.values()) | set(state.enemy_locks)
        unavailable = banned | picked

        pool_has_role = any(c.champ_id in self.personal.pool_ids for c in self.by_role[state.my_role])
        overall_cands = [
            c for c in self.by_role[state.my_role]
            if c.champ_id not in unavailable
            and not (state.mastery_mode == "POOL" and pool_has_role and c.champ_id not in self.personal.pool_ids)
        ]

        # Resolve user's pre-game shortlist (bypassing POOL filter so explicit picks always get scored)
        resolved_focus, unmatched_focus = self.resolve_focus_champions(state.focus_names)
        unavailable_focus: List[str] = []
        focus_cids: List[int] = []
        extra_cands: Dict[int, ChampProfile] = {}
        off_role_cids: Set[int] = set()
        overall_cid_set = {c.champ_id for c in overall_cands}

        for cid, disp_name in resolved_focus:
            if cid in banned:
                unavailable_focus.append(f"{disp_name} (Banned)")
                continue
            if cid in picked:
                unavailable_focus.append(f"{disp_name} (Picked)")
                continue
            focus_cids.append(cid)
            if cid not in overall_cid_set:
                if (state.my_role, cid) in self.champions:
                    extra_cands[cid] = self.champions[(state.my_role, cid)]
                elif cid in self.cid_best_profile:
                    # Off-role fallback: use mechanical profile with higher LCB uncertainty
                    base_p = self.cid_best_profile[cid]
                    extra_cands[cid] = replace(base_p, role=state.my_role, base_games=15)
                    off_role_cids.add(cid)

        all_to_score = overall_cands + list(extra_cands.values())
        scored_map = self._score_candidates(all_to_score, state, unavailable, off_role_cids)

        overall_recs = sorted(
            [scored_map[c.champ_id] for c in overall_cands if c.champ_id in scored_map],
            key=lambda x: x["lcb_wr"], reverse=True,
        )[:top_overall]

        best_overall_lcb = overall_recs[0]["lcb_wr"] if overall_recs else None
        best_overall_cid = overall_recs[0]["champ_id"] if overall_recs else None

        shortlist_recs = []
        for cid in focus_cids:
            if cid in scored_map:
                rec = dict(scored_map[cid])
                if best_overall_lcb is not None:
                    rec["delta_vs_best"] = round(rec["lcb_wr"] - best_overall_lcb, 2)
                    rec["is_best_overall"] = (cid == best_overall_cid) or (rec["delta_vs_best"] >= 0.0)
                else:
                    rec["delta_vs_best"] = 0.0
                    rec["is_best_overall"] = False
                shortlist_recs.append(rec)

        shortlist_recs.sort(key=lambda x: x["lcb_wr"], reverse=True)
        return {
            "shortlist": shortlist_recs,
            "overall": overall_recs,
            "unavailable_focus": unavailable_focus,
            "unmatched_focus": unmatched_focus,
        }

    def recommend_bans(self, state: DraftState, top_k: int = 5) -> List[dict]:
        """Protects your Shortlist picks if active; otherwise protects your Top 3 overall picks."""
        bundle = self.recommend_with_shortlist(state, top_overall=3)
        protected = bundle["shortlist"][:4] if bundle["shortlist"] else bundle["overall"][:3]
        if not protected:
            return []
        is_shortlist = bool(bundle["shortlist"])
        unavailable = state.ally_bans | state.enemy_bans
        elo_key = str(state.elo_tier).upper()
        ban_cands = sorted(
            [c for c in self.by_role[state.my_role] if c.champ_id not in unavailable],
            key=lambda c: self._eff_play_rate(c, elo_key), reverse=True
        )[:22]

        scored_bans = []
        for b in ban_cands:
            b_pr = self._eff_play_rate(b, elo_key)
            lift = sum(
                max(0.0, -self._delta_and_var(state.my_role, p["champ_id"], state.my_role, b.champ_id, True)[0])
                * b_pr * state.n_enemy_pool
                for p in protected
            )
            meta_threat = max(0.0, logit(b.base_wr) + b.elo_wr_delta.get(elo_key, 0.0)) * b_pr
            ev_gain = (lift / len(protected) + 0.6 * meta_threat) * 25.0
            tag = "shortlist" if is_shortlist else "top picks"
            scored_bans.append({
                "champ_id": b.champ_id, "champ": b.name,
                "ev_gain_wr": round(ev_gain, 2), "play_rate": round(b_pr * 100, 1),
                "reason": f"Protects {tag}: {', '.join(p['champ'] for p in protected[:3])}",
            })
        scored_bans.sort(key=lambda x: x["ev_gain_wr"], reverse=True)
        return scored_bans[:top_k]

    def compute_swap_advisor(self, state: DraftState) -> Dict[str, float]:
        top_enemies = self.solve_enemy_permutations(state, min_prob=0.05)[0][0]
        open_allies = [r for r in ROLES if r not in state.locked_allies]
        open_enemies = [r for r in ROLES if r not in top_enemies]
        unavail = state.ally_bans | state.enemy_bans | set(state.locked_allies.values()) | set(state.enemy_locks)
        exposures = {}
        for role in open_allies:
            enemy_exp = sum(MATCHUP_HEATMAP[role][er] * (MATCHUP_HEATMAP[er][role] / 5.0) for er in open_enemies)
            ally_exp = 0.0
            for ar in open_allies:
                if ar == role:
                    continue
                h = state.hovered_allies.get(ar, 0)
                hover_discount = (1.0 - 0.7 * state.hover_confidence) if (h > 0 and h not in unavail) else 1.0
                ally_exp += (SYNERGY_HEATMAP[role][ar] / 4.0) * hover_discount
            exposures[role] = round(enemy_exp + 0.65 * ally_exp, 3)
        return dict(sorted(exposures.items(), key=lambda kv: kv[1], reverse=True))