import asyncio, json, os, sqlite3, sys, time
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from core.lcu_watcher import parse_lcu_session, watch_lcu_loop
from core.scoring_engine import DraftState, PersonalProfile
from data.db_loader import load_engine_from_db
from data.ddragon_assets import DDragonAssets
from ui.overlay import DraftOverlay

SETTINGS_PATH = "config/user_settings.json"
DB_PATH = "data/patch_matrix.db"


class LCUWorker(QThread):
    phase_sig = Signal(str)
    session_sig = Signal(dict, tuple)
    personal_sig = Signal(object)
    eog_sig = Signal(dict)
    rank_sig = Signal(str)

    def run(self):
        asyncio.run(
            watch_lcu_loop(
                self.phase_sig.emit,
                self.session_sig.emit,
                self.personal_sig.emit,
                self.eog_sig.emit,
                self.rank_sig.emit,
            )
        )


class DraftEngineApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        with open(SETTINGS_PATH) as f:
            self.settings = json.load(f)

        self.assets = DDragonAssets(DB_PATH)
        self.app.aboutToQuit.connect(self.assets.stop)

        self.engine = load_engine_from_db(DB_PATH, SETTINGS_PATH)
        if self.settings.get("custom_pool_ids"):
            self.engine.personal.pool_ids = set(self.settings["custom_pool_ids"])

        self.detected_elo = None
        self.last_state, self.last_recs, self.pending_lock_cid = None, [], 0
        self.overlay = DraftOverlay(self.settings, self.assets)
        self.overlay.move(40, 80)
        self.overlay.settings_changed.connect(self.on_settings_changed)
        self.overlay.size_changed.connect(self.on_size_changed)
        self._setup_tray()

        self.assets.start_icon_prefetch()

        self.worker = LCUWorker()
        self.worker.phase_sig.connect(self.on_phase)
        self.worker.session_sig.connect(self.on_session)
        self.worker.personal_sig.connect(self.on_personal)
        self.worker.eog_sig.connect(self.on_eog)
        self.worker.rank_sig.connect(self.on_rank_detected)
        self.worker.start()

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self.app.style().standardIcon(QStyle.SP_ComputerIcon), self.app)
        menu = QMenu()
        menu.addAction("Show / Hide Overlay", lambda: self.overlay.setVisible(not self.overlay.isVisible()))
        menu.addAction("Demo: Pick Phase (Dual Scroll + Shortlist)", lambda: self.run_demo("PICK"))
        menu.addAction("Demo: Ban Phase (Protects Shortlist)", lambda: self.run_demo("BAN"))
        menu.addSeparator()
        menu.addAction("Quit DraftEngine", self.app.quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip(f"DraftEngine v4 -- {self.settings.get('elo_tier', 'EMERALD')} (Stacker v2)")
        self.tray.show()

    def on_phase(self, phase: str):
        if phase == "ChampSelect":
            self.overlay.show()
        elif phase == "InProgress":
            self._log_pregame_prediction()
            self.overlay.hide()
        elif phase in ("None", "Lobby", "EndOfGame"):
            self.overlay.hide()

    def on_personal(self, personal: PersonalProfile):
        if self.settings.get("custom_pool_ids"):
            personal.pool_ids = set(self.settings["custom_pool_ids"])
        self.engine.personal = personal

    def on_rank_detected(self, tier: str):
        tier = tier.upper()
        if not tier:
            return
        self.detected_elo = tier
        if self.settings.get("elo_tier", "").upper() != tier:
            self.settings["elo_tier"] = tier
            try:
                with open(SETTINGS_PATH, "w") as f:
                    json.dump(self.settings, f, indent=2)
            except Exception:
                pass
        self.tray.setToolTip(f"DraftEngine v4 -- Auto-Rank: {tier} (Stacker v2)")
        if self.last_state and self.last_state.elo_tier != tier:
            self.last_state.elo_tier = tier
            self._refresh_ui()

    def on_session(self, raw_session: dict, role_prefs: tuple):
        state = parse_lcu_session(raw_session, self.settings, role_prefs, detected_elo=self.detected_elo)
        if not state:
            return
        local_cell = raw_session.get("localPlayerCellId")
        for m in raw_session.get("myTeam", []):
            if m.get("cellId") == local_cell and int(m.get("championId", 0)) > 0:
                self.pending_lock_cid = int(m["championId"])
        self.last_state = state
        self._refresh_ui()
        if not self.overlay.isVisible():
            self.overlay.show()

    def _log_pregame_prediction(self):
        if not (self.last_state and self.last_recs and self.pending_lock_cid and os.path.exists(DB_PATH)):
            return
        top1 = self.last_recs[0]["champ_id"]
        top2 = self.last_recs[1]["champ_id"] if len(self.last_recs) > 1 else top1
        rec = next((r for r in self.last_recs if r["champ_id"] == self.pending_lock_cid), self.last_recs[0])
        try:
            con = sqlite3.connect(DB_PATH, timeout=30.0)
            con.execute("""
                INSERT INTO personal_calibration_log
                (timestamp, my_role, locked_cid, top1_cid, top2_cid, followed_top2, pred_wr, pred_std)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                int(time.time()), self.last_state.my_role, self.pending_lock_cid,
                top1, top2, int(self.pending_lock_cid in (top1, top2)),
                rec["est_wr"] / 100.0, rec["sigma_wr"] / 100.0,
            ))
            con.commit(); con.close()
        except Exception:
            pass

    def on_eog(self, eog_data: dict):
        if not os.path.exists(DB_PATH):
            return
        try:
            won = int(bool(eog_data.get("localPlayer", {}).get("stats", {}).get("WIN", 0)))
            con = sqlite3.connect(DB_PATH, timeout=30.0)
            con.execute("""
                UPDATE personal_calibration_log SET actual_win=?, game_id=?
                WHERE id = (SELECT MAX(id) FROM personal_calibration_log WHERE actual_win = -1)
            """, (won, str(eog_data.get("gameId", ""))))
            con.commit(); con.close()
        except Exception:
            pass

    def on_size_changed(self, w: int, h: int):
        self.settings["overlay_size"] = [w, h]
        with open(SETTINGS_PATH, "w") as f:
            json.dump(self.settings, f, indent=2)

    def on_settings_changed(self, new_settings: dict):
        self.settings = new_settings
        with open(SETTINGS_PATH, "w") as f:
            json.dump(self.settings, f, indent=2)
        if self.last_state:
            self.last_state.mastery_mode = self.settings["mastery_mode"]
            self.last_state.xp_stall_mode = self.settings["xp_stall_mode"]
            self.last_state.focus_names = list(self.settings.get("focus_champs", []))
            self._refresh_ui()

    def _refresh_ui(self):
        bundle = self.engine.recommend_with_shortlist(self.last_state, top_overall=10)
        seen, merged = set(), []
        for r in (bundle["shortlist"] + bundle["overall"]):
            if r["champ_id"] not in seen:
                seen.add(r["champ_id"])
                merged.append(r)
        self.last_recs = merged

        bans = self.engine.recommend_bans(self.last_state, top_k=8) if self.last_state.phase == "BAN" else []
        exposures = self.engine.compute_swap_advisor(self.last_state)
        alerts = self.engine.top_ban_inference_alerts(self.last_state)
        onnx_on = bool(self.engine.onnx_net and self.engine.onnx_net.active)
        self.overlay.render_state(self.last_state, bundle, bans, exposures, alerts, onnx_on)

    def run_demo(self, phase="PICK"):
        focus = self.settings.get("focus_champs") or ["Ahri", "Orianna", "Yasuo", "Aurelion Sol"]
        self.overlay.set_focus_champs(focus)
        self.last_state = DraftState(
            my_role="MID", my_pick_index=2, is_blue_side=True, phase=phase,
            elo_tier=self.detected_elo or self.settings.get("elo_tier", "EMERALD"),
            role_status="PRIMARY",
            locked_allies={"JGL": 102},
            hovered_allies={"SUP": 111},
            enemy_locks=[79, 516],
            ally_bans={800},
            enemy_bans={54},
            mastery_mode=self.settings["mastery_mode"],
            mastery_weight=self.settings["mastery_weight"],
            lcb_kappa=self.settings.get("lcb_kappa", 0.75),
            hover_confidence=self.settings.get("hover_confidence", 0.75),
            xp_stall_mode=self.settings["xp_stall_mode"],
            focus_names=focus,
        )
        self._refresh_ui()
        self.overlay.show()

    def run(self):
        sys.exit(self.app.exec())


if __name__ == "__main__":
    DraftEngineApp().run()