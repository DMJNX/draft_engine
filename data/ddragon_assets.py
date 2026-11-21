import json, os, sqlite3, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap

CACHE_JSON = "data/ddragon_cache.json"
ICON_DIR = "data/icons"
CACHE_TTL_SEC = 3 * 86400  # Refresh champion catalog every 3 days


class _IconDownloadWorker(QThread):
    """Downloads missing champion PNGs to disk on background threads (never touches QPixmap)."""
    downloaded = Signal(int)

    def __init__(self, version: str, items: list[dict]):
        super().__init__()
        self.version = version
        self.items = items
        self._stop = False

    def stop(self):
        self._stop = True

    def _fetch_one(self, item: dict):
        if self._stop:
            return
        key = item["key"]
        dest = os.path.join(ICON_DIR, f"{key}.png")
        if os.path.exists(dest) and os.path.getsize(dest) > 200:
            return
        url = f"https://ddragon.leagueoflegends.com/cdn/{self.version}/img/champion/{key}.png"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200 and len(r.content) > 200:
                tmp = f"{dest}.tmp"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, dest)
                self.downloaded.emit(item["cid"])
        except Exception:
            pass

    def run(self):
        os.makedirs(ICON_DIR, exist_ok=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(self._fetch_one, it) for it in self.items]
            for fut in futures:
                if self._stop:
                    break
                try:
                    fut.result(timeout=10)
                except Exception:
                    pass


class DDragonAssets(QObject):
    """Loads full League champion catalog + caches rounded champion icons locally."""
    icon_ready = Signal(int)

    def __init__(self, db_path: str = "data/patch_matrix.db"):
        super().__init__()
        self.db_path = db_path
        self.version = ""
        self.all_champions: list[dict] = []
        self.by_cid: dict[int, dict] = {}
        self.by_name_lower: dict[str, dict] = {}
        self._pixmap_cache: dict[tuple[int, int, bool], QPixmap] = {}
        self._worker: _IconDownloadWorker | None = None
        self._load_catalog()

    def _load_catalog(self):
        os.makedirs("data", exist_ok=True)
        # 1. Try fresh local cache first
        if os.path.exists(CACHE_JSON):
            try:
                with open(CACHE_JSON, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("champions") and (time.time() - cached.get("ts", 0)) < CACHE_TTL_SEC:
                    self._populate(cached.get("version", ""), cached["champions"])
                    return
            except Exception:
                pass

        # 2. Fetch live catalog from Riot Data Dragon (covers every champion in the game)
        try:
            versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=6).json()
            latest = versions[0]
            data = requests.get(
                f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json", timeout=8
            ).json()["data"]
            champs = sorted(
                [{"cid": int(v["key"]), "key": k, "name": v["name"]} for k, v in data.items()],
                key=lambda x: x["name"].lower(),
            )
            with open(CACHE_JSON, "w", encoding="utf-8") as f:
                json.dump({"version": latest, "ts": int(time.time()), "champions": champs}, f)
            self._populate(latest, champs)
            return
        except Exception:
            pass

        # 3. Offline fallback: stale cache or SQLite champ_meta
        if os.path.exists(CACHE_JSON):
            try:
                with open(CACHE_JSON, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                self._populate(cached.get("version", ""), cached.get("champions", []))
                return
            except Exception:
                pass

        if os.path.exists(self.db_path):
            try:
                con = sqlite3.connect(self.db_path, timeout=5.0)
                rows = con.execute("SELECT cid, key_name, display_name FROM champ_meta").fetchall()
                con.close()
                champs = sorted(
                    [{"cid": int(r[0]), "key": r[1], "name": r[2]} for r in rows],
                    key=lambda x: x["name"].lower(),
                )
                self._populate("", champs)
            except Exception:
                pass

    def _populate(self, version: str, champs: list[dict]):
        self.version = version
        self.all_champions = champs
        self.by_cid = {c["cid"]: c for c in champs}
        self.by_name_lower = {c["name"].lower(): c for c in champs}

    def start_icon_prefetch(self):
        if not self.version or not self.all_champions:
            return
        missing = [
            c for c in self.all_champions
            if not os.path.exists(os.path.join(ICON_DIR, f"{c['key']}.png"))
        ]
        if not missing:
            return
        self._worker = _IconDownloadWorker(self.version, missing)
        self._worker.downloaded.connect(self._on_icon_downloaded)
        self._worker.start()

    def _on_icon_downloaded(self, cid: int):
        # Clear any placeholder cache entries for this champ and notify UI
        for k in [k for k in self._pixmap_cache if k[0] == cid]:
            del self._pixmap_cache[k]
        self.icon_ready.emit(cid)

    def stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(1500)

    def lookup_by_name(self, name: str) -> dict | None:
        return self.by_name_lower.get(name.strip().lower())

    def get_pixmap(self, cid: int, size: int = 28, rounded: bool = True) -> QPixmap:
        cache_key = (cid, size, rounded)
        if cache_key in self._pixmap_cache:
            return self._pixmap_cache[cache_key]

        meta = self.by_cid.get(cid)
        if meta:
            path = os.path.join(ICON_DIR, f"{meta['key']}.png")
            if os.path.exists(path):
                raw = QPixmap(path)
                if not raw.isNull():
                    scaled = raw.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                    if not rounded:
                        self._pixmap_cache[cache_key] = scaled
                        return scaled
                    out = QPixmap(size, size)
                    out.fill(Qt.transparent)
                    p = QPainter(out)
                    p.setRenderHint(QPainter.Antialiasing)
                    clip = QPainterPath()
                    clip.addRoundedRect(0, 0, size, size, 5, 5)
                    p.setClipPath(clip)
                    p.drawPixmap(0, 0, scaled)
                    p.end()
                    self._pixmap_cache[cache_key] = out
                    return out

        # Clean dark monogram placeholder while PNG downloads
        ph = QPixmap(size, size)
        ph.fill(Qt.transparent)
        p = QPainter(ph)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#1f293b"))
        p.setPen(QColor("#3a4860"))
        p.drawRoundedRect(0, 0, size - 1, size - 1, 5, 5)
        initials = (meta["name"][:2].upper() if meta else "?")
        p.setPen(QColor("#c89b3c"))
        f = QFont("Segoe UI", max(7, int(size * 0.36)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(ph.rect(), Qt.AlignCenter, initials)
        p.end()
        return ph

    def get_icon(self, cid: int, size: int = 20) -> QIcon:
        return QIcon(self.get_pixmap(cid, size=size, rounded=True))