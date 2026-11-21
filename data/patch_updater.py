import json, os, queue, random, sqlite3, sys, threading, time
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("RIOT_API_KEY", "").strip()

REGIONS = [
    ("EUW", "euw1", "europe"),
    ("NA",  "na1",  "americas"),
    ("KR",  "kr",   "asia"),
]

ALL_TIERS_AND_DIVS = [
    ("MASTER", "I"),
    ("DIAMOND", "I"), ("DIAMOND", "II"), ("DIAMOND", "III"), ("DIAMOND", "IV"),
    ("EMERALD", "I"), ("EMERALD", "II"), ("EMERALD", "III"), ("EMERALD", "IV"),
    ("PLATINUM", "I"), ("PLATINUM", "II"), ("PLATINUM", "III"), ("PLATINUM", "IV"),
    ("GOLD", "I"), ("GOLD", "II"), ("GOLD", "III"), ("GOLD", "IV"),
    ("SILVER", "I"), ("SILVER", "II"), ("SILVER", "III"), ("SILVER", "IV"),
    ("BRONZE", "I"), ("BRONZE", "II"), ("BRONZE", "III"), ("BRONZE", "IV"),
    ("IRON", "I"), ("IRON", "II"), ("IRON", "III"), ("IRON", "IV"),
]

TIER_PRESETS = {
    "ALL":  {"IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"},
    "LOW":  {"IRON", "BRONZE", "SILVER", "GOLD"},
    "MID":  {"SILVER", "GOLD", "PLATINUM", "EMERALD"},
    "HIGH": {"DIAMOND", "MASTER"},
}

ROLE_MAP = {"TOP": "TOP", "JUNGLE": "JGL", "MIDDLE": "MID", "BOTTOM": "BOT", "UTILITY": "SUP"}
TAG_MOBILITY = {"Assassin": 0.90, "Fighter": 0.55, "Marksman": 0.35, "Mage": 0.25, "Tank": 0.35, "Support": 0.30}
LOOKBACK_DAYS = 115
MASTERY_CACHE_TTL = 14 * 86400  # 14 days
PLAYERS_PER_DIV_ROUND = 8       # Round-robin batch size so no single tier starves the others


def resolve_tier_targets(spec: str) -> list[tuple[str, str]]:
    key = (spec or "ALL").strip().upper()
    if key in TIER_PRESETS:
        allowed = TIER_PRESETS[key]
    else:
        allowed = {tok.strip() for tok in key.split(",") if tok.strip()}
    targets = [(t, d) for t, d in ALL_TIERS_AND_DIVS if t in allowed]
    return targets if targets else list(ALL_TIERS_AND_DIVS)


class RateLimitedRiotClient:
    def __init__(self, api_key: str, stop_event: threading.Event, delay: float = 1.25):
        self.delay = delay
        self.stop_event = stop_event
        self.last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": api_key})

    def get(self, url: str, params: dict = None):
        retries = 0
        while not self.stop_event.is_set():
            wait = self.delay - (time.monotonic() - self.last_call)
            if wait > 0 and self.stop_event.wait(wait):
                return None
            self.last_call = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=15)
            except requests.RequestException:
                retries += 1
                if retries > 5 or self.stop_event.wait(min(30, 3 * retries)):
                    return None
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                if self.stop_event.wait(int(resp.headers.get("Retry-After", 10)) + 1):
                    return None
                continue
            if resp.status_code in (401, 403):
                print(f"\n[AUTH ERROR {resp.status_code}] API key rejected/expired. Stopping cleanly...")
                self.stop_event.set()
                return None
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                retries += 1
                if retries > 4 or self.stop_event.wait(5 * retries):
                    return None
                continue
            return None
        return None


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS champ_meta (
            cid INT PRIMARY KEY, key_name TEXT, display_name TEXT,
            difficulty INT DEFAULT 5, attack_range REAL DEFAULT 175.0, mobility_prior REAL DEFAULT 0.4
        );
        CREATE TABLE IF NOT EXISTS rune_meta (rune_id INT PRIMARY KEY, name TEXT);
        CREATE TABLE IF NOT EXISTS seen_matches (match_id TEXT PRIMARY KEY, patch TEXT);
        CREATE TABLE IF NOT EXISTS patch_totals (patch TEXT PRIMARY KEY, match_count INT);

        CREATE TABLE IF NOT EXISTS puuid_mastery_cache (
            puuid TEXT, cid INT, points INT, updated_ts INT,
            PRIMARY KEY (puuid, cid)
        );
        CREATE TABLE IF NOT EXISTS mastery_subsample (
            match_id TEXT, role TEXT, cid INT, mastery_pts INT, win INT,
            PRIMARY KEY (match_id, role, cid)
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY, patch TEXT, region TEXT, elo_tier TEXT,
            duration_sec INT, blue_win INT, first_tower_blue INT,
            grubs_blue INT, grubs_red INT, soul_blue INT, soul_red INT,
            baron_blue INT, baron_red INT, bans_blue TEXT, bans_red TEXT
        );
        CREATE TABLE IF NOT EXISTS match_participants (
            match_id TEXT, team_id INT, role TEXT, cid INT, win INT,
            keystone_id INT, substyle_id INT, spell1_id INT, spell2_id INT,
            gd15 REAL, xpd15 REAL, phys_share REAL, mag_share REAL, true_share REAL,
            mit_dpm REAL, heal_shield_pm REAL,
            PRIMARY KEY (match_id, team_id, role)
        );

        CREATE TABLE IF NOT EXISTS champ_patch_stats (
            patch TEXT, role TEXT, cid INT,
            wins INT, games INT,
            b0_wins INT, b0_games INT, b1_wins INT, b1_games INT,
            b2_wins INT, b2_games INT, b3_wins INT, b3_games INT,
            blue_wins INT, blue_games INT,
            phys_dpm REAL, mag_dpm REAL, true_dpm REAL, mit_dpm REAL,
            cc_pm REAL, cs_pm REAL, heal_shield_pm REAL, gpm REAL,
            PRIMARY KEY (patch, role, cid)
        );
        CREATE TABLE IF NOT EXISTS matchup_patch_stats (
            patch TEXT, r1 TEXT, c1 INT, r2 TEXT, c2 INT, wins INT, games INT,
            PRIMARY KEY (patch, r1, c1, r2, c2)
        );
        CREATE TABLE IF NOT EXISTS synergy_patch_stats (
            patch TEXT, r1 TEXT, c1 INT, r2 TEXT, c2 INT, wins INT, games INT,
            PRIMARY KEY (patch, r1, c1, r2, c2)
        );
        CREATE TABLE IF NOT EXISTS ban_pick_stats (
            patch TEXT, is_same_team INT, banned_cid INT, role TEXT, picked_cid INT, count INT,
            PRIMARY KEY (patch, is_same_team, banned_cid, role, picked_cid)
        );
        CREATE TABLE IF NOT EXISTS setup_stats (
            patch TEXT, role TEXT, cid INT, enemy_cid INT,
            keystone_id INT, substyle_id INT, spell_pair TEXT,
            wins INT, games INT,
            PRIMARY KEY (patch, role, cid, enemy_cid, keystone_id, substyle_id, spell_pair)
        );
        CREATE TABLE IF NOT EXISTS elo_champ_stats (
            patch TEXT, elo_tier TEXT, role TEXT, cid INT, wins INT, games INT,
            PRIMARY KEY (patch, elo_tier, role, cid)
        );
        CREATE TABLE IF NOT EXISTS personal_calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INT, my_role TEXT, locked_cid INT,
            top1_cid INT, top2_cid INT, followed_top2 INT,
            pred_wr REAL, pred_std REAL DEFAULT 0.0,
            actual_win INT DEFAULT -1, game_id TEXT DEFAULT ''
        );
    """)
    con.commit()
    return con


def sync_ddragon(con: sqlite3.Connection) -> str:
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=10).json()
    latest = versions[0]
    champs = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json", timeout=10
    ).json()["data"]
    rows = []
    for k, v in champs.items():
        tags = v.get("tags", [])
        mob = sum(TAG_MOBILITY.get(t, 0.35) for t in tags) / max(1, len(tags))
        ms_bonus = max(0.0, (float(v.get("stats", {}).get("movespeed", 335)) - 325.0) / 60.0) * 0.2
        rows.append((
            int(v["key"]), k, v["name"],
            int(v.get("info", {}).get("difficulty", 5)),
            float(v.get("stats", {}).get("attackrange", 175.0)),
            min(1.0, mob + ms_bonus),
        ))
    con.executemany("INSERT OR REPLACE INTO champ_meta VALUES (?,?,?,?,?,?)", rows)

    runes = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/runesReforged.json", timeout=10
    ).json()
    rune_rows = []
    for tree in runes:
        rune_rows.append((int(tree["id"]), tree["name"]))
        for slot in tree.get("slots", []):
            for r in slot.get("runes", []):
                rune_rows.append((int(r["id"]), r["name"]))
    con.executemany("INSERT OR REPLACE INTO rune_meta VALUES (?,?)", rune_rows)
    con.commit()
    return ".".join(latest.split(".")[:2])


def _extract_setup(p: dict):
    styles = p.get("perks", {}).get("styles", [])
    keystone = styles[0]["selections"][0]["perk"] if len(styles) > 0 and styles[0].get("selections") else 0
    substyle = styles[1]["style"] if len(styles) > 1 else 0
    s1, s2 = sorted([int(p.get("summoner1Id", 4)), int(p.get("summoner2Id", 14))])
    return int(keystone), int(substyle), s1, s2, f"{s1}-{s2}"


def _extract_tempo_proxy(p: dict, opp: dict, dur_min: float):
    ch = p.get("challenges", {})
    opp_ch = opp.get("challenges", {}) if opp else {}
    cs_lead = float(ch.get("maxCsAdvantageOnLaneOpponent", 0.0))
    lvl_lead = float(ch.get("maxLevelLeadLaneOpponent", 0.0))
    gpm_diff = float(ch.get("goldPerMinute", p.get("goldEarned", 0) / dur_min)) - float(
        opp_ch.get("goldPerMinute", opp.get("goldEarned", 0) / dur_min if opp else 0.0)
    )
    gd15 = float(ch.get("laningPhaseGoldExpAdvantage", 0.0)) * 450.0 + cs_lead * 18.0 + gpm_diff * 5.0
    xpd15 = lvl_lead * 320.0 + float(ch.get("earlyLaningPhaseGoldExpAdvantage", 0.0)) * 300.0
    return max(-4000.0, min(4000.0, gd15)), max(-3000.0, min(3000.0, xpd15))


def _sample_match_masteries(
    client: RateLimitedRiotClient,
    platform: str,
    info: dict,
    mastery_cache: dict,
    cache_lock: threading.Lock,
) -> tuple[list[tuple], list[tuple]]:
    now_ts = int(time.time())
    subsample_rows = []
    new_cache_rows = []
    for p in info.get("participants", []):
        role = ROLE_MAP.get(p.get("teamPosition"))
        puuid = p.get("puuid")
        cid = int(p.get("championId", 0))
        if not (role and puuid and cid > 0):
            continue
        with cache_lock:
            cached = mastery_cache.get((puuid, cid))
        if cached and (now_ts - cached[1]) < MASTERY_CACHE_TTL:
            pts = cached[0]
        else:
            url = f"https://{platform}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{cid}"
            res = client.get(url)
            pts = int(res.get("championPoints", 0)) if isinstance(res, dict) else 0
            with cache_lock:
                mastery_cache[(puuid, cid)] = (pts, now_ts)
            new_cache_rows.append((puuid, cid, pts, now_ts))
        subsample_rows.append((role, cid, pts, int(bool(p.get("win", False)))))
    return subsample_rows, new_cache_rows


def region_worker(
    tag, platform, regional, tier_targets, seen_ids, seen_lock, mastery_cache, cache_lock,
    subsample_rate, out_queue, stop_event
):
    client = RateLimitedRiotClient(API_KEY, stop_event, delay=1.25)
    rng = random.Random()
    start_time = int(time.time()) - (LOOKBACK_DAYS * 24 * 60 * 60)
    seen_puuids = set()
    master_cache = None

    for page in range(1, 25):
        if stop_event.is_set():
            break
        targets_order = list(tier_targets)
        rng.shuffle(targets_order)
        for tier, div in targets_order:
            if stop_event.is_set():
                break
            if tier == "MASTER":
                if master_cache is None:
                    res = client.get(f"https://{platform}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5")
                    master_cache = res.get("entries", []) if isinstance(res, dict) else []
                    rng.shuffle(master_cache)
                entries = master_cache
            else:
                entries = client.get(
                    f"https://{platform}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{div}",
                    params={"page": page},
                ) or []
                rng.shuffle(entries)

            crawled_here = 0
            for entry in entries:
                if stop_event.is_set() or crawled_here >= PLAYERS_PER_DIV_ROUND:
                    break
                puuid = entry.get("puuid")
                if not puuid or puuid in seen_puuids:
                    continue
                seen_puuids.add(puuid)
                crawled_here += 1

                new_mids = []
                for offset in (0, 100):
                    if stop_event.is_set():
                        break
                    mids = client.get(
                        f"https://{regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids",
                        params={"queue": 420, "type": "ranked", "startTime": start_time, "start": offset, "count": 100},
                    )
                    if not mids:
                        break
                    with seen_lock:
                        for mid in mids:
                            if mid not in seen_ids:
                                seen_ids.add(mid)
                                new_mids.append(mid)
                    if len(mids) < 100:
                        break

                for mid in new_mids:
                    if stop_event.is_set():
                        break
                    data = client.get(f"https://{regional}.api.riotgames.com/lol/match/v5/matches/{mid}")
                    if not data or "info" not in data:
                        continue
                    info = data["info"]
                    m_rows, c_rows = [], []
                    if info.get("gameDuration", 0) >= 600 and rng.random() < subsample_rate:
                        m_rows, c_rows = _sample_match_masteries(client, platform, info, mastery_cache, cache_lock)

                    while not stop_event.is_set():
                        try:
                            out_queue.put((tag, tier, mid, info, m_rows, c_rows), timeout=0.5)
                            break
                        except queue.Full:
                            continue


def store_match(con: sqlite3.Connection, tag: str, elo_tier: str, mid: str, info: dict, m_rows: list, c_rows: list):
    if c_rows:
        con.executemany("INSERT OR REPLACE INTO puuid_mastery_cache VALUES (?,?,?,?)", c_rows)

    dur_sec = int(info.get("gameDuration", 0))
    if dur_sec < 600:
        con.execute("INSERT OR IGNORE INTO seen_matches VALUES (?, 'SKIP')", (mid,))
        return None

    teams = {100: {}, 200: {}}
    for p in info.get("participants", []):
        role = ROLE_MAP.get(p.get("teamPosition"))
        tid = p.get("teamId")
        if role and tid in teams and role not in teams[tid]:
            teams[tid][role] = p

    if len(teams[100]) != 5 or len(teams[200]) != 5:
        con.execute("INSERT OR IGNORE INTO seen_matches VALUES (?, 'SKIP')", (mid,))
        return None

    patch = ".".join(info.get("gameVersion", "0.0").split(".")[:2])
    cur = con.execute("INSERT OR IGNORE INTO seen_matches VALUES (?, ?)", (mid, patch))
    if cur.rowcount == 0:
        return None

    if m_rows:
        con.executemany(
            "INSERT OR IGNORE INTO mastery_subsample VALUES (?,?,?,?,?)",
            [(mid, r, c, pts, w) for (r, c, pts, w) in m_rows],
        )

    dur_min = dur_sec / 60.0
    b_idx = 0 if dur_min < 25 else (1 if dur_min < 30 else (2 if dur_min < 35 else 3))

    team_objs = {t["teamId"]: t for t in info.get("teams", [])}
    t100, t200 = team_objs.get(100, {}), team_objs.get(200, {})
    o100, o200 = t100.get("objectives", {}), t200.get("objectives", {})
    bans_100 = [int(b["championId"]) for b in t100.get("bans", []) if b.get("championId", 0) > 0]
    bans_200 = [int(b["championId"]) for b in t200.get("bans", []) if b.get("championId", 0) > 0]

    con.execute("INSERT INTO patch_totals VALUES (?, 1) ON CONFLICT(patch) DO UPDATE SET match_count = match_count + 1", (patch,))
    con.execute("""
        INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        mid, patch, tag, elo_tier, dur_sec,
        int(bool(t100.get("win", False))),
        int(bool(o100.get("tower", {}).get("first", False))),
        int(o100.get("horde", {}).get("kills", 0)), int(o200.get("horde", {}).get("kills", 0)),
        int(o100.get("dragon", {}).get("kills", 0) >= 4), int(o200.get("dragon", {}).get("kills", 0) >= 4),
        int(o100.get("baron", {}).get("kills", 0) >= 1), int(o200.get("baron", {}).get("kills", 0) >= 1),
        json.dumps(bans_100), json.dumps(bans_200),
    ))

    for tid, eid, team_bans, enemy_bans in ((100, 200, bans_100, bans_200), (200, 100, bans_200, bans_100)):
        allies_map, enemies_map = teams[tid], teams[eid]
        for r1, p1 in allies_map.items():
            c1, win = int(p1["championId"]), int(bool(p1["win"]))
            ks, sub, s1, s2, spair = _extract_setup(p1)
            opp = enemies_map.get(r1)
            gd15, xpd15 = _extract_tempo_proxy(p1, opp, dur_min)

            phys = float(p1.get("physicalDamageDealtToChampions", 0.0))
            mag = float(p1.get("magicDamageDealtToChampions", 0.0))
            true_d = float(p1.get("trueDamageDealtToChampions", 0.0))
            tot_d = max(1.0, phys + mag + true_d)
            mit_dpm = (float(p1.get("damageSelfMitigated", 0.0)) + float(p1.get("totalDamageTaken", 0.0))) / dur_min
            hs_pm = (
                float(p1.get("totalHealsOnTeammates", 0.0))
                + float(p1.get("totalDamageShieldedOnTeammates", 0.0))
                + 0.35 * float(p1.get("totalHeal", 0.0))
            ) / dur_min
            gpm = float(p1.get("goldEarned", 0.0)) / dur_min

            con.execute("""
                INSERT OR IGNORE INTO match_participants VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                mid, tid, r1, c1, win, ks, sub, s1, s2,
                gd15, xpd15, phys / tot_d, mag / tot_d, true_d / tot_d, mit_dpm, hs_pm,
            ))

            b_vals = [0] * 8
            b_vals[b_idx * 2] = win
            b_vals[b_idx * 2 + 1] = 1
            is_blue = int(tid == 100)

            con.execute("""
                INSERT INTO champ_patch_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(patch, role, cid) DO UPDATE SET
                    wins=wins+excluded.wins, games=games+1,
                    b0_wins=b0_wins+excluded.b0_wins, b0_games=b0_games+excluded.b0_games,
                    b1_wins=b1_wins+excluded.b1_wins, b1_games=b1_games+excluded.b1_games,
                    b2_wins=b2_wins+excluded.b2_wins, b2_games=b2_games+excluded.b2_games,
                    b3_wins=b3_wins+excluded.b3_wins, b3_games=b3_games+excluded.b3_games,
                    blue_wins=blue_wins+excluded.blue_wins, blue_games=blue_games+excluded.blue_games,
                    phys_dpm=phys_dpm+excluded.phys_dpm, mag_dpm=mag_dpm+excluded.mag_dpm,
                    true_dpm=true_dpm+excluded.true_dpm, mit_dpm=mit_dpm+excluded.mit_dpm,
                    cc_pm=cc_pm+excluded.cc_pm, cs_pm=cs_pm+excluded.cs_pm,
                    heal_shield_pm=heal_shield_pm+excluded.heal_shield_pm, gpm=gpm+excluded.gpm
            """, (
                patch, r1, c1, win, 1, *b_vals, win * is_blue, is_blue,
                phys / dur_min, mag / dur_min, true_d / dur_min, mit_dpm,
                float(p1.get("timeCCingOthers", 0.0)) / dur_min,
                float(p1.get("totalMinionsKilled", 0.0)) / dur_min,
                hs_pm, gpm,
            ))

            con.execute("""
                INSERT INTO elo_champ_stats VALUES (?,?,?,?,?,1)
                ON CONFLICT(patch, elo_tier, role, cid) DO UPDATE SET wins=wins+excluded.wins, games=games+1
            """, (patch, elo_tier, r1, c1, win))

            for target_enemy in (0, int(opp["championId"]) if opp else -1):
                if target_enemy >= 0:
                    con.execute("""
                        INSERT INTO setup_stats VALUES (?,?,?,?,?,?,?,?,1)
                        ON CONFLICT(patch, role, cid, enemy_cid, keystone_id, substyle_id, spell_pair)
                        DO UPDATE SET wins=wins+excluded.wins, games=games+1
                    """, (patch, r1, c1, target_enemy, ks, sub, spair, win))

            for b_cid in team_bans:
                con.execute("INSERT INTO ban_pick_stats VALUES (?,1,?,?,?,1) ON CONFLICT(patch, is_same_team, banned_cid, role, picked_cid) DO UPDATE SET count=count+1", (patch, b_cid, r1, c1))
            for b_cid in enemy_bans:
                con.execute("INSERT INTO ban_pick_stats VALUES (?,0,?,?,?,1) ON CONFLICT(patch, is_same_team, banned_cid, role, picked_cid) DO UPDATE SET count=count+1", (patch, b_cid, r1, c1))

            for r2, p2 in enemies_map.items():
                con.execute("INSERT INTO matchup_patch_stats VALUES (?,?,?,?,?,?,1) ON CONFLICT(patch, r1, c1, r2, c2) DO UPDATE SET wins=wins+excluded.wins, games=games+1", (patch, r1, c1, r2, int(p2["championId"]), win))
            for r2, p2 in allies_map.items():
                if r1 != r2:
                    con.execute("INSERT INTO synergy_patch_stats VALUES (?,?,?,?,?,?,1) ON CONFLICT(patch, r1, c1, r2, c2) DO UPDATE SET wins=wins+excluded.wins, games=games+1", (patch, r1, c1, r2, int(p2["championId"]), win))

    return patch


def crawl_parallel(max_hours: float = 9.0, tier_spec: str = "ALL", db_path: str = "data/patch_matrix.db"):
    with open("config/user_settings.json") as f:
        subsample_rate = float(json.load(f).get("mastery_subsample_rate", 0.10))

    tier_targets = resolve_tier_targets(tier_spec)
    tier_names = sorted({t for t, _ in tier_targets})

    con = init_db(db_path)
    latest_patch = sync_ddragon(con)
    if not API_KEY:
        print("Synced Data Dragon names, ranges, mobility & runes. Set RIOT_API_KEY in .env to crawl.")
        con.close()
        return

    test = requests.get("https://euw1.api.riotgames.com/lol/status/v4/platform-data", headers={"X-Riot-Token": API_KEY}, timeout=10)
    if test.status_code != 200:
        print(f"API Key pre-flight failed ({test.status_code}): {test.text[:160]}")
        con.close()
        return

    seen_ids = {r[0] for r in con.execute("SELECT match_id FROM seen_matches")}
    mastery_cache = {(r[0], r[1]): (r[2], r[3]) for r in con.execute("SELECT puuid, cid, points, updated_ts FROM puuid_mastery_cache")}
    existing_valid = con.execute("SELECT COALESCE(SUM(match_count), 0) FROM patch_totals").fetchone()[0]
    print(f"Latest patch: {latest_patch} | DB valid: {existing_valid:,} | Cached PUUID masteries: {len(mastery_cache):,}")
    print(f"3 parallel workers (EUW/NA/KR) | Tiers: {', '.join(tier_names)} | {int(subsample_rate*100)}% mastery subsample | Max {max_hours:.1f}h (Ctrl+C saves).\n")

    seen_lock, cache_lock = threading.Lock(), threading.Lock()
    stop_event = threading.Event()
    out_queue: queue.Queue = queue.Queue(maxsize=250)

    threads = [
        threading.Thread(
            target=region_worker,
            args=(tag, platform, regional, tier_targets, seen_ids, seen_lock, mastery_cache, cache_lock, subsample_rate, out_queue, stop_event),
            daemon=True,
        )
        for tag, platform, regional in REGIONS
    ]
    for t in threads:
        t.start()

    start_ts = time.monotonic()
    deadline = start_ts + max_hours * 3600.0
    added, uncommitted, sampled = 0, 0, 0
    reg_counts = {tag: 0 for tag, _, _ in REGIONS}
    tier_counts = {}

    try:
        while time.monotonic() < deadline and not stop_event.is_set():
            try:
                tag, tier, mid, info, m_rows, c_rows = out_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue
            patch = store_match(con, tag, tier, mid, info, m_rows, c_rows)
            uncommitted += 1
            if patch:
                added += 1
                reg_counts[tag] += 1
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                if m_rows:
                    sampled += 1
            if uncommitted >= 25:
                con.commit()
                uncommitted = 0
            if added > 0 and added % 50 == 0 and patch:
                hrs = max((time.monotonic() - start_ts) / 3600.0, 1e-4)
                rem = max(0, int((deadline - time.monotonic()) / 60))
                t_str = " ".join(f"{k[:3]}:{v}" for k, v in sorted(tier_counts.items()))
                print(f"[+{added:,} | {int(added/hrs):,}/hr | {rem}m left] EUW:{reg_counts['EUW']} NA:{reg_counts['NA']} KR:{reg_counts['KR']} | {t_str} | Sub:{sampled}")
    except KeyboardInterrupt:
        print("\nCtrl+C detected -- saving cleanly...")
    finally:
        stop_event.set()
        while not out_queue.empty():
            try:
                tag, tier, mid, info, m_rows, c_rows = out_queue.get_nowait()
                if store_match(con, tag, tier, mid, info, m_rows, c_rows):
                    added += 1
            except queue.Empty:
                break
        con.commit()
        tot = con.execute("SELECT COALESCE(SUM(match_count), 0) FROM patch_totals").fetchone()[0]
        con.close()
        for t in threads:
            t.join(timeout=3.0)
        print(f"Saved cleanly. +{added:,} matches (DB total: {tot:,} valid matches).")


if __name__ == "__main__":
    hrs = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
    t_spec = sys.argv[2] if len(sys.argv) > 2 else "ALL"
    crawl_parallel(max_hours=hrs, tier_spec=t_spec)