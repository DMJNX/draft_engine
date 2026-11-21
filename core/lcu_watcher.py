import asyncio, base64, json, math, os, ssl, time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
import aiohttp

from core.scoring_engine import ROLES, DraftState, PersonalProfile, logit

LOCKFILE_PATHS = [
    r"C:\Riot Games\League of Legends\lockfile",
    os.path.expandvars(r"%LOCALAPPDATA%\Riot Games\League of Legends\lockfile"),
]
LCU_POS_MAP = {"top": "TOP", "jungle": "JGL", "middle": "MID", "bottom": "BOT", "utility": "SUP"}
TIER_NORMALIZE = {
    "IRON": "IRON", "BRONZE": "BRONZE", "SILVER": "SILVER", "GOLD": "GOLD",
    "PLATINUM": "PLATINUM", "EMERALD": "EMERALD", "DIAMOND": "DIAMOND",
    "MASTER": "MASTER", "GRANDMASTER": "MASTER", "CHALLENGER": "MASTER",
}


def read_lockfile() -> Optional[Tuple[int, str]]:
    for path in LOCKFILE_PATHS:
        if os.path.exists(path):
            try:
                _, _, port, password, _ = open(path, "r").read().strip().split(":")
                return int(port), base64.b64encode(f"riot:{password}".encode()).decode()
            except Exception:
                return None
    return None


def parse_lcu_session(data: dict, settings: dict, role_prefs: Tuple[str, str], detected_elo: Optional[str] = None) -> Optional[DraftState]:
    local_cell = data.get("localPlayerCellId")
    my_team = data.get("myTeam", [])
    their_team = data.get("theirTeam", [])
    if local_cell is None or not my_team:
        return None

    my_role, my_pick_index, is_blue = "MID", 0, True
    locked_allies: Dict[str, int] = {}
    hovered_allies: Dict[str, int] = {}
    my_cells = {m.get("cellId") for m in my_team}

    for idx, member in enumerate(my_team):
        role = LCU_POS_MAP.get(member.get("assignedPosition", "").lower())
        cid = int(member.get("championId", 0))
        intent = int(member.get("championPickIntent", 0))
        if member.get("cellId") == local_cell:
            my_role = role or "MID"
            my_pick_index = idx
            is_blue = int(member.get("team", 1)) == 1
        else:
            if not role:
                used = set(locked_allies) | set(hovered_allies) | {my_role}
                open_r = [r for r in ROLES if r not in used]
                role = open_r[0] if open_r else "TOP"
            if cid > 0:
                locked_allies[role] = cid
            elif intent > 0:
                hovered_allies[role] = intent  # Soft hover prior (75/25)

    enemy_locks: List[int] = [int(m.get("championId", 0)) for m in their_team if int(m.get("championId", 0)) > 0]

    bans_obj = data.get("bans", {})
    ally_bans = {int(c) for c in bans_obj.get("myTeamBans", []) if int(c) > 0}
    enemy_bans = {int(c) for c in bans_obj.get("theirTeamBans", []) if int(c) > 0}

    in_ban_phase = False
    for group in data.get("actions", []):
        for act in group:
            if act.get("type") == "ban":
                if act.get("isInProgress") and act.get("actorCellId") == local_cell:
                    in_ban_phase = True
                if act.get("completed") and int(act.get("championId", 0)) > 0:
                    (ally_bans if act.get("actorCellId") in my_cells else enemy_bans).add(int(act["championId"]))

    pref1, pref2 = role_prefs
    if not pref1 or pref1 == "FILL" or my_role == pref1:
        role_status = "PRIMARY"
    elif my_role == pref2:
        role_status = "SECONDARY"
    else:
        role_status = "AUTOFILL"

    elo_tier = (detected_elo or settings.get("elo_tier", "EMERALD")).upper()

    return DraftState(
        my_role=my_role, my_pick_index=my_pick_index, is_blue_side=is_blue,
        phase="BAN" if in_ban_phase else "PICK",
        elo_tier=elo_tier, role_status=role_status,
        locked_allies=locked_allies, hovered_allies=hovered_allies,
        enemy_locks=enemy_locks, ally_bans=ally_bans, enemy_bans=enemy_bans,
        mastery_mode=settings.get("mastery_mode", "POOL"),
        mastery_weight=float(settings.get("mastery_weight", 0.75)),
        lcb_kappa=float(settings.get("lcb_kappa", 0.75)),
        hover_confidence=float(settings.get("hover_confidence", 0.75)),
        n_enemy_pool=int(settings.get("n_enemy_pool", 7)),
        expectimax_depth=int(settings.get("expectimax_depth", 1)),
        xp_stall_mode=bool(settings.get("xp_stall_mode", False)),
        autofill_protection=bool(settings.get("autofill_protection", True)),
        focus_names=list(settings.get("focus_champs", [])),
    )


async def fetch_local_rank(session: aiohttp.ClientSession, port: int, ssl_ctx) -> Optional[str]:
    """Read-only LCU call to auto-detect the local summoner's Solo/Duo (or Flex / prev-season) rank tier."""
    try:
        async with session.get(f"https://127.0.0.1:{port}/lol-ranked/v1/current-ranked-stats", ssl=ssl_ctx) as r:
            if r.status != 200:
                return None
            data = await r.json()
        qmap = data.get("queueMap") or {}
        solo = qmap.get("RANKED_SOLO_5x5") or {}
        flex = qmap.get("RANKED_FLEX_SR") or {}
        candidates = [
            solo.get("tier"),
            solo.get("previousSeasonEndTier"),
            flex.get("tier"),
            flex.get("previousSeasonEndTier"),
            data.get("highestPreviousSeasonEndTier"),
        ]
        for raw in candidates:
            norm = TIER_NORMALIZE.get(str(raw or "").upper())
            if norm:
                return norm
    except Exception:
        pass
    return None


async def fetch_personal_profile(session: aiohttp.ClientSession, port: int, ssl_ctx) -> PersonalProfile:
    prof = PersonalProfile()
    try:
        async with session.get(f"https://127.0.0.1:{port}/lol-champion-mastery/v1/local-player/champion-mastery", ssl=ssl_ctx) as r:
            if r.status == 200:
                for item in await r.json():
                    cid, pts, lvl = int(item.get("championId", 0)), int(item.get("championPoints", 0)), int(item.get("championLevel", 0))
                    prof.mastery_points[cid] = pts
                    if lvl >= 5 or pts >= 21600:
                        prof.pool_ids.add(cid)
    except Exception:
        pass

    try:
        url = f"https://127.0.0.1:{port}/lol-match-history/v1/products/lol/current-summoner/matches?begIndex=0&endIndex=80"
        async with session.get(url, ssl=ssl_ctx) as r:
            if r.status != 200:
                return prof
            games = (await r.json()).get("games", {}).get("games", [])
    except Exception:
        return prof

    cutoff_45d = time.time() * 1000.0 - (45 * 86400 * 1000.0)
    champ_rec = defaultdict(lambda: [0, 0])
    early_wl, late_wl = [0, 0], [0, 0]

    for g in games:
        dur = int(g.get("gameDuration", 0))
        if dur < 600:
            continue
        p0 = (g.get("participants") or [{}])[0]
        cid, win = int(p0.get("championId", 0)), int(bool(p0.get("stats", {}).get("win", False)))
        if cid <= 0:
            continue
        if int(g.get("gameCreation", 0)) >= cutoff_45d:
            prof.recent_games_45d[cid] = prof.recent_games_45d.get(cid, 0) + 1
            champ_rec[cid][0] += win; champ_rec[cid][1] += 1
            if champ_rec[cid][1] >= 4:
                prof.pool_ids.add(cid)
        if dur <= 1560:
            early_wl[0] += win; early_wl[1] += 1
        elif dur >= 1860:
            late_wl[0] += win; late_wl[1] += 1

    for cid, (w, n) in champ_rec.items():
        prof.mastery_logit[cid] = (n / (n + 20.0)) * logit((w + 2.0) / (n + 4.0))
    if early_wl[1] >= 5 and late_wl[1] >= 5:
        prof.duration_slope = max(-0.20, min(0.20, ((late_wl[0]+3.0)/(late_wl[1]+6.0)) - ((early_wl[0]+3.0)/(early_wl[1]+6.0))))
    return prof


async def fetch_lobby_role_prefs(session: aiohttp.ClientSession, port: int, ssl_ctx) -> Tuple[str, str]:
    try:
        async with session.get(f"https://127.0.0.1:{port}/lol-lobby/v2/lobby", ssl=ssl_ctx) as r:
            if r.status == 200:
                m = (await r.json()).get("localMember", {})
                return (
                    LCU_POS_MAP.get(str(m.get("firstPositionPreference", "")).lower(), ""),
                    LCU_POS_MAP.get(str(m.get("secondPositionPreference", "")).lower(), ""),
                )
    except Exception:
        pass
    return "", ""


async def watch_lcu_loop(on_phase, on_session, on_personal, on_eog, on_rank=None):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    while True:
        creds = read_lockfile()
        if not creds:
            on_phase("Disconnected")
            await asyncio.sleep(3)
            continue
        port, token = creds
        headers = {"Authorization": f"Basic {token}"}
        role_prefs = ("", "")
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                on_personal(await fetch_personal_profile(session, port, ssl_ctx))
                role_prefs = await fetch_lobby_role_prefs(session, port, ssl_ctx)
                if on_rank:
                    rank = await fetch_local_rank(session, port, ssl_ctx)
                    if rank:
                        on_rank(rank)
                async with session.ws_connect(f"wss://127.0.0.1:{port}/", ssl=ssl_ctx) as ws:
                    await ws.send_json([5, "OnJsonApiEvent_lol-gameflow_v1_gameflow-phase"])
                    await ws.send_json([5, "OnJsonApiEvent_lol-champ-select_v1_session"])
                    await ws.send_json([5, "OnJsonApiEvent_lol-end-of-game_v1_eog-stats-block"])
                    on_phase("Connected")
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT or not msg.data:
                            continue
                        payload = json.loads(msg.data)
                        if len(payload) != 3:
                            continue
                        ev, body = payload[1], payload[2]
                        if ev == "OnJsonApiEvent_lol-gameflow_v1_gameflow-phase":
                            phase = body.get("data", "None")
                            if phase in ("Lobby", "Matchmaking", "ChampSelect"):
                                role_prefs = await fetch_lobby_role_prefs(session, port, ssl_ctx)
                                if on_rank:
                                    rank = await fetch_local_rank(session, port, ssl_ctx)
                                    if rank:
                                        on_rank(rank)
                            on_phase(phase)
                        elif ev == "OnJsonApiEvent_lol-champ-select_v1_session":
                            if body.get("eventType") == "Delete":
                                on_phase("None")
                            elif body.get("data"):
                                on_session(body["data"], role_prefs)
                        elif ev == "OnJsonApiEvent_lol-end-of-game_v1_eog-stats-block" and body.get("data"):
                            on_eog(body["data"])
        except Exception:
            await asyncio.sleep(2.5)