from collections import defaultdict
from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCompleter, QFrame, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QPushButton, QScrollArea, QSizeGrip, QSplitter,
    QVBoxLayout, QWidget,
)

RESIZE_MARGIN = 8

STYLE = """
QWidget#Root {
    background-color: rgba(13, 17, 25, 244); border: 1px solid #2b3547; border-radius: 10px;
    color: #e6edf3; font-family: 'Segoe UI', sans-serif;
}
QLabel#Title   { font-size: 13px; font-weight: 700; color: #c89b3c; letter-spacing: 0.5px; }
QLabel#Sub     { font-size: 10px; color: #8b949e; }
QLabel#SecHdr  { font-size: 10px; font-weight: 700; color: #c89b3c; letter-spacing: 0.8px; padding-top: 2px; }
QLabel#SecSub  { font-size: 10px; font-weight: 700; color: #8b949e; letter-spacing: 0.8px; padding-top: 2px; }
QLabel#Footer  { font-size: 9px; color: #566376; }
QLabel#SwapBanner {
    background-color: rgba(200, 155, 60, 0.14); border: 1px solid rgba(200, 155, 60, 0.45);
    border-radius: 6px; padding: 4px 8px; font-size: 10px; color: #f0d28c;
}
QLabel#AlertBanner {
    background-color: rgba(88, 166, 255, 0.14); border: 1px solid rgba(88, 166, 255, 0.45);
    border-radius: 6px; padding: 4px 8px; font-size: 10px; color: #9ecbff;
}
QLineEdit {
    background-color: #141b29; border: 1px solid #303b4e; border-radius: 5px;
    color: #e6edf3; font-size: 10.5px; padding: 5px 8px;
}
QLineEdit:focus { border-color: #c89b3c; }
QPushButton {
    background-color: #1b2333; border: 1px solid #303b4e; border-radius: 5px;
    color: #9aa7b8; font-size: 10px; font-weight: 600; padding: 3px 7px;
}
QPushButton:checked { background-color: #c89b3c; border-color: #e3bc65; color: #0d1119; }
QFrame#Chip {
    background-color: #1b2638; border: 1px solid #3a4a66; border-radius: 12px;
}
QPushButton#ChipX {
    background-color: transparent; border: none; color: #9aa7b8;
    font-size: 12px; font-weight: 700; padding: 0px 2px;
}
QPushButton#ChipX:hover { color: #ff7b72; }
QFrame#Card { background-color: #141b29; border: 1px solid #232e42; border-radius: 7px; }
QFrame#ShortlistBest { background-color: #16221e; border: 1px solid #2e6b4f; border-radius: 7px; }

QScrollArea, QWidget#ScrollContent { background: transparent; border: none; }
QScrollBar:vertical {
    background: transparent; width: 6px; margin: 2px 0px 2px 2px;
}
QScrollBar::handle:vertical {
    background: #303b4e; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #c89b3c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent; height: 0px;
}

QSplitter::handle:vertical {
    background-color: #263144; height: 5px; margin: 3px 28px; border-radius: 2px;
}
QSplitter::handle:vertical:hover { background-color: #c89b3c; }
QSizeGrip { background: transparent; width: 14px; height: 14px; }
"""

POPUP_STYLE = """
QListView {
    background-color: #121824; color: #e6edf3; border: 1px solid #c89b3c;
    border-radius: 6px; font-family: 'Segoe UI', sans-serif; font-size: 11px;
    outline: none; padding: 2px;
}
QListView::item { padding: 4px 6px; border-radius: 4px; }
QListView::item:selected, QListView::item:hover {
    background-color: #26334a; color: #ffffff;
}
"""


class FlowLayout(QLayout):
    """Wrapping horizontal layout so locked-in champion chips flow cleanly across rows."""
    def __init__(self, parent=None, spacing: int = 5):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x, y, line_h = rect.x(), rect.y(), 0
        for item in self._items:
            w, h = item.sizeHint().width(), item.sizeHint().height()
            next_x = x + w + self._spacing
            if next_x - self._spacing > rect.right() and line_h > 0:
                x = rect.x()
                y = y + line_h + self._spacing
                next_x = x + w + self._spacing
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_h = max(line_h, h)
        return y + line_h - rect.y()


class DraftOverlay(QWidget):
    settings_changed = Signal(dict)
    size_changed = Signal(int, int)

    def __init__(self, initial_settings: dict, assets=None):
        super().__init__()
        self.settings = dict(initial_settings)
        self.assets = assets
        self.focus_champs: list[str] = list(self.settings.get("focus_champs", []))

        # Drag & frameless edge-resize state
        self._drag_pos: QPoint | None = None
        self._resize_edges: set[str] = set()
        self._resize_start_global: QPoint | None = None
        self._resize_start_geom: QRect | None = None
        self._user_moved_splitter = False

        # Live icon registries
        self._completer_items: dict[int, QStandardItem] = {}
        self._chip_icon_slots: dict[int, list[tuple[QLabel, int]]] = defaultdict(list)
        self._card_icon_slots: dict[int, list[tuple[QLabel, int]]] = defaultdict(list)

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_Hover, True)
        self.setMinimumSize(360, 340)

        saved_sz = self.settings.get("overlay_size", [415, 650])
        self.resize(max(360, int(saved_sz[0])), max(340, int(saved_sz[1])))

        # Debounce persisting window size so dragging stays silky smooth
        self._size_save_timer = QTimer(self)
        self._size_save_timer.setSingleShot(True)
        self._size_save_timer.setInterval(450)
        self._size_save_timer.timeout.connect(lambda: self.size_changed.emit(self.width(), self.height()))

        root = QWidget(self)
        root.setObjectName("Root")
        root.setStyleSheet(STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        self.main_lay = QVBoxLayout(root)
        self.main_lay.setContentsMargins(12, 10, 12, 6)
        self.main_lay.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        self.lbl_title = QLabel("DRAFTENGINE v3  ·  STANDBY")
        self.lbl_title.setObjectName("Title")
        btn_hide = QPushButton("×")
        btn_hide.setFixedSize(22, 22)
        btn_hide.clicked.connect(self.hide)
        hdr.addWidget(self.lbl_title); hdr.addStretch(); hdr.addWidget(btn_hide)
        self.main_lay.addLayout(hdr)

        self.lbl_status = QLabel("Waiting for Champion Select...")
        self.lbl_status.setObjectName("Sub")
        self.main_lay.addWidget(self.lbl_status)

        # Toggles Row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        self.mastery_btns = {}
        for mode in ("OFF", "POOL", "ON"):
            b = QPushButton(f"M:{mode}")
            b.setCheckable(True)
            b.setChecked(self.settings.get("mastery_mode") == mode)
            b.clicked.connect(lambda _, m=mode: self._set_mastery(m))
            self.mastery_btns[mode] = b
            ctrl.addWidget(b)
        ctrl.addStretch()
        self.btn_xp = QPushButton("XP Stall")
        self.btn_xp.setCheckable(True)
        self.btn_xp.setChecked(self.settings.get("xp_stall_mode", False))
        self.btn_xp.clicked.connect(self._toggle_xp)
        ctrl.addWidget(self.btn_xp)
        self.main_lay.addLayout(ctrl)

        # Searchable Shortlist Picker Row + Autocomplete Dropdown
        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        self.focus_input = QLineEdit()
        self.focus_input.setPlaceholderText("Add to shortlist (type champion name)...")
        self.focus_input.returnPressed.connect(self._on_search_enter)
        self._setup_completer()

        btn_clear_focus = QPushButton("Clear")
        btn_clear_focus.setFixedWidth(44)
        btn_clear_focus.clicked.connect(self._clear_focus)
        search_row.addWidget(self.focus_input)
        search_row.addWidget(btn_clear_focus)
        self.main_lay.addLayout(search_row)

        # Locked-in Shortlist Chips Container
        self.chips_container = QWidget()
        self.chips_layout = FlowLayout(self.chips_container, spacing=5)
        self.chips_container.hide()
        self.main_lay.addWidget(self.chips_container)

        # Alerts & Swap Advisor
        self.lbl_alerts = QLabel("")
        self.lbl_alerts.setObjectName("AlertBanner")
        self.lbl_alerts.setWordWrap(True)
        self.lbl_alerts.hide()
        self.main_lay.addWidget(self.lbl_alerts)

        self.lbl_swap = QLabel("")
        self.lbl_swap.setObjectName("SwapBanner")
        self.lbl_swap.setWordWrap(True)
        self.lbl_swap.hide()
        self.main_lay.addWidget(self.lbl_swap)

        # Split Dual-Scroll Viewport (Shortlist on top, Overall/Bans on bottom)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        self.shortlist_pane, self.lbl_shortlist_hdr, self.shortlist_scroll, self.shortlist_lay = self._make_scroll_pane(
            "YOUR SHORTLIST (BEST FIRST)", "SecHdr"
        )
        self.overall_pane, self.lbl_overall_hdr, self.overall_scroll, self.overall_lay = self._make_scroll_pane(
            "BEST OVERALL", "SecSub"
        )

        self.splitter.addWidget(self.shortlist_pane)
        self.splitter.addWidget(self.overall_pane)
        self.shortlist_pane.hide()
        self.main_lay.addWidget(self.splitter, 1)

        # Footer with subtle resize cue + native bottom-right QSizeGrip
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        lbl_hint = QLabel("Drag edges or corner to resize · Drag divider to adjust split")
        lbl_hint.setObjectName("Footer")
        footer.addWidget(lbl_hint)
        footer.addStretch()
        grip = QSizeGrip(self)
        grip.setToolTip("Drag to resize overlay")
        footer.addWidget(grip, 0, Qt.AlignRight | Qt.AlignBottom)
        self.main_lay.addLayout(footer)

        if self.assets:
            self.assets.icon_ready.connect(self._on_icon_ready)
        self._rebuild_chips()

    def _make_scroll_pane(self, header_text: str, header_obj_name: str):
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        hdr = QLabel(header_text)
        hdr.setObjectName(header_obj_name)
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("ScrollContent")
        cards_lay = QVBoxLayout(content)
        cards_lay.setContentsMargins(0, 0, 4, 0)
        cards_lay.setSpacing(5)
        cards_lay.addStretch(1)

        scroll.setWidget(content)
        lay.addWidget(scroll, 1)
        return pane, hdr, scroll, cards_lay

    def _on_splitter_moved(self, _pos: int, _index: int):
        self._user_moved_splitter = True

    # ------------------------------------------------------------------
    # Autocomplete + Lock-In Chips Logic
    # ------------------------------------------------------------------
    def _setup_completer(self):
        model = QStandardItemModel(self)
        if self.assets:
            for c in self.assets.all_champions:
                item = QStandardItem(self.assets.get_icon(c["cid"], 20), c["name"])
                model.appendRow(item)
                self._completer_items[c["cid"]] = item

        self.completer = QCompleter(model, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setMaxVisibleItems(8)

        popup = self.completer.popup()
        popup.setStyleSheet(POPUP_STYLE)
        popup.setIconSize(QSize(20, 20))
        self.completer.activated[str].connect(self._on_completer_selected)
        self.focus_input.setCompleter(self.completer)

    def _on_completer_selected(self, text: str):
        self._lock_focus_champ(text)
        QTimer.singleShot(0, self.focus_input.clear)

    def _on_search_enter(self):
        raw = self.focus_input.text().strip()
        if not raw:
            return
        if self.assets and self.assets.lookup_by_name(raw):
            chosen = self.assets.lookup_by_name(raw)["name"]
        elif self.completer.completionCount() > 0:
            chosen = self.completer.currentCompletion()
        else:
            chosen = raw
        if chosen:
            self._lock_focus_champ(chosen)
        QTimer.singleShot(0, self.focus_input.clear)

    def _lock_focus_champ(self, name: str):
        canonical = name.strip()
        if self.assets:
            meta = self.assets.lookup_by_name(canonical)
            if meta:
                canonical = meta["name"]
        if not canonical or any(x.lower() == canonical.lower() for x in self.focus_champs):
            return
        self.focus_champs.append(canonical)
        self._commit_focus_change()

    def _remove_focus_champ(self, name: str):
        self.focus_champs = [c for c in self.focus_champs if c.lower() != name.lower()]
        self._commit_focus_change()

    def _clear_focus(self):
        self.focus_input.clear()
        if self.focus_champs:
            self.focus_champs = []
            self._user_moved_splitter = False
            self._commit_focus_change()

    def set_focus_champs(self, names: list[str]):
        self.focus_champs = list(names)
        self._rebuild_chips()

    def _commit_focus_change(self):
        self.settings["focus_champs"] = list(self.focus_champs)
        self._rebuild_chips()
        self.settings_changed.emit(self.settings)

    def _rebuild_chips(self):
        self._chip_icon_slots.clear()
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not self.focus_champs:
            self.chips_container.hide()
            return

        for name in self.focus_champs:
            chip = QFrame()
            chip.setObjectName("Chip")
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(5, 2, 5, 2)
            cl.setSpacing(4)

            if self.assets:
                meta = self.assets.lookup_by_name(name)
                if meta:
                    icon_lbl = QLabel()
                    icon_lbl.setFixedSize(18, 18)
                    icon_lbl.setPixmap(self.assets.get_pixmap(meta["cid"], 18, rounded=True))
                    self._chip_icon_slots[meta["cid"]].append((icon_lbl, 18))
                    cl.addWidget(icon_lbl)

            txt = QLabel(name)
            txt.setStyleSheet("font-size: 10.5px; font-weight: 600; color: #e6edf3;")
            cl.addWidget(txt)

            btn_x = QPushButton("×")
            btn_x.setObjectName("ChipX")
            btn_x.setFixedSize(15, 15)
            btn_x.setCursor(Qt.PointingHandCursor)
            btn_x.clicked.connect(lambda _, n=name: self._remove_focus_champ(n))
            cl.addWidget(btn_x)

            self.chips_layout.addWidget(chip)

        self.chips_container.show()
        self.chips_container.updateGeometry()

    # ------------------------------------------------------------------
    # Live Background Icon Pop-In
    # ------------------------------------------------------------------
    def _on_icon_ready(self, cid: int):
        if not self.assets:
            return
        if cid in self._completer_items:
            self._completer_items[cid].setIcon(self.assets.get_icon(cid, 20))
        for lbl, sz in self._chip_icon_slots.get(cid, []):
            lbl.setPixmap(self.assets.get_pixmap(cid, sz, rounded=True))
        for lbl, sz in self._card_icon_slots.get(cid, []):
            lbl.setPixmap(self.assets.get_pixmap(cid, sz, rounded=True))

    # ------------------------------------------------------------------
    # Toggles & Dual-Scroll Card Rendering
    # ------------------------------------------------------------------
    def _set_mastery(self, mode: str):
        self.settings["mastery_mode"] = mode
        for m, btn in self.mastery_btns.items():
            btn.setChecked(m == mode)
        self.settings_changed.emit(self.settings)

    def _toggle_xp(self):
        self.settings["xp_stall_mode"] = self.btn_xp.isChecked()
        self.settings_changed.emit(self.settings)

    def _clear_vbox(self, lay: QVBoxLayout):
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _make_icon_label(self, cid: int, size: int) -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(size, size)
        if self.assets:
            lbl.setPixmap(self.assets.get_pixmap(cid, size, rounded=True))
            self._card_icon_slots[cid].append((lbl, size))
        return lbl

    def _build_pick_card(self, rank: int, r: dict, is_shortlist: bool = False) -> QFrame:
        card = QFrame()
        card.setObjectName("ShortlistBest" if (is_shortlist and rank == 1) else "Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(9, 6, 9, 6)
        cl.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(7)
        if self.assets:
            top.addWidget(self._make_icon_label(r["champ_id"], 28))

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        prefix = f"★ #{rank}" if (is_shortlist and rank == 1) else f"#{rank}"
        name = QLabel(f"{prefix}  {r['champ']}")
        name.setStyleSheet("font-size: 13px; font-weight: 700; color: #ffffff;")
        title_col.addWidget(name)

        if is_shortlist and "delta_vs_best" in r:
            if r.get("is_best_overall"):
                cmp_lbl = QLabel("✓ Matches #1 Overall Pick")
                cmp_lbl.setStyleSheet("font-size: 9.5px; color: #3fb950; font-weight: 700;")
            else:
                gap = r["delta_vs_best"]
                gcol = "#e3b341" if gap >= -1.5 else "#ff7b72"
                cmp_lbl = QLabel(f"Gap vs #1 Overall: {gap:+.1f}% LCB")
                cmp_lbl.setStyleSheet(f"font-size: 9.5px; color: {gcol}; font-weight: 600;")
            title_col.addWidget(cmp_lbl)

        top.addLayout(title_col)
        top.addStretch()

        col = "#3fb950" if r["lcb_wr"] >= 50.5 else ("#e3b341" if r["lcb_wr"] >= 48.0 else "#f85149")
        wr = QLabel(f"LCB {r['lcb_wr']:.1f}%\n({r['est_wr']:.1f}±{r['sigma_wr']:.1f}%)")
        wr.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        wr.setStyleSheet(f"font-size: 10.5px; font-weight: 700; color: {col};")
        top.addWidget(wr)
        cl.addLayout(top)

        item_str = (" · [" + ", ".join(r["item_flags"]) + "]") if r["item_flags"] else ""
        setup_lbl = QLabel(f"⚡ {r['setup']}{item_str}")
        setup_lbl.setStyleSheet("font-size: 10px; color: #79c0ff; font-weight: 600;")
        setup_lbl.setWordWrap(True)
        cl.addWidget(setup_lbl)

        b = r["breakdown_wr"]
        risk_tag = f"CtrRisk {r['lane_counter_prob']:.0f}%" if r["lane_counter_prob"] > 0 else "Lane Shown"
        otp_str = f" OTP{b['otp_debias']:+.1f}" if abs(b["otp_debias"]) >= 0.2 else ""
        det = QLabel(
            f"Base {b['base']:+.1f}{otp_str} | Ctr {b['matchups']:+.1f} | Syn {b['synergy']:+.1f} | "
            f"ExpMax {b['blind_risk']:+.1f} | Comp {b['comp']:+.1f} | M {b['mastery']:+.1f} · {risk_tag}"
        )
        det.setStyleSheet("font-size: 9.5px; color: #9aa7b8;")
        det.setWordWrap(True)
        cl.addWidget(det)

        if r["flags"]:
            warn = QLabel("⚠ " + " · ".join(r["flags"]))
            warn.setStyleSheet("font-size: 10px; color: #f85149; font-weight: 600;")
            cl.addWidget(warn)
        return card

    def render_state(self, state, bundle: dict, bans: list, exposures: dict, ban_alerts: list, onnx_active: bool):
        side = "BLUE" if state.is_blue_side else "RED"
        af_tag = f" · [{state.role_status}]" if state.role_status != "PRIMARY" else ""
        nn_tag = "L1+5xEnsemble+L3" if onnx_active else "L1(Bayes/SVD)+L3"
        hovers_tag = f" | Hovers:{len(state.hovered_allies)}" if state.hovered_allies else ""
        self.lbl_title.setText(f"{state.phase} · {side} {state.my_role} (#{state.my_pick_index + 1}){af_tag}")
        self.lbl_status.setText(f"{nn_tag} | LCB κ={state.lcb_kappa:.2f}{hovers_tag}")

        if ban_alerts:
            self.lbl_alerts.setText("BAN INFERENCE: " + " · ".join(ban_alerts))
            self.lbl_alerts.show()
        else:
            self.lbl_alerts.hide()

        if state.my_role in exposures and len(exposures) > 1:
            top_r, top_e = next(iter(exposures.items()))
            my_e = exposures[state.my_role]
            if top_r != state.my_role and (top_e - my_e) > 0.25:
                self.lbl_swap.setText(f"SWAP ADVISOR: {top_r} exposure ({top_e:.2f}) > {state.my_role} ({my_e:.2f}). Offer swap.")
            else:
                self.lbl_swap.setText(f"HOLD SLOT: {state.my_role} exposure is high ({my_e:.2f}).")
            self.lbl_swap.show()
        else:
            self.lbl_swap.hide()

        self._card_icon_slots.clear()
        self._clear_vbox(self.shortlist_lay)
        self._clear_vbox(self.overall_lay)

        if state.phase == "BAN" and bans:
            self.shortlist_pane.hide()
            self.lbl_overall_hdr.setText(f"RECOMMENDED BANS ({len(bans)})")
            for rank, b in enumerate(bans, 1):
                card = QFrame(); card.setObjectName("Card")
                cl = QVBoxLayout(card); cl.setContentsMargins(9, 6, 9, 6); cl.setSpacing(2)
                top = QHBoxLayout(); top.setSpacing(7)
                if self.assets:
                    top.addWidget(self._make_icon_label(b["champ_id"], 24))
                t1 = QLabel(f"BAN #{rank}  {b['champ']}")
                t1.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff7b72;")
                t2 = QLabel(f"+{b['ev_gain_wr']:.2f}% EV ({b['play_rate']}% PR)")
                t2.setStyleSheet("font-size: 11px; font-weight: 700; color: #f0d28c;")
                top.addWidget(t1); top.addStretch(); top.addWidget(t2)
                cl.addLayout(top)
                sub = QLabel(b["reason"]); sub.setStyleSheet("font-size: 10px; color: #9aa7b8;")
                cl.addWidget(sub)
                self.overall_lay.addWidget(card)
            self.overall_lay.addStretch(1)
            self.overall_scroll.verticalScrollBar().setValue(0)
            return

        shortlist = bundle.get("shortlist", [])
        overall = bundle.get("overall", [])
        unavail = bundle.get("unavailable_focus", [])
        unmatched = bundle.get("unmatched_focus", [])
        has_focus_mode = bool(state.focus_names)

        if has_focus_mode:
            self.lbl_shortlist_hdr.setText(f"YOUR SHORTLIST ({len(shortlist)} AVAILABLE)")
            for rank, r in enumerate(shortlist, 1):
                self.shortlist_lay.addWidget(self._build_pick_card(rank, r, is_shortlist=True))
            if unavail:
                u_lbl = QLabel("Unavailable: " + " · ".join(unavail))
                u_lbl.setStyleSheet("font-size: 10px; color: #f85149;")
                u_lbl.setWordWrap(True)
                self.shortlist_lay.addWidget(u_lbl)
            if unmatched:
                m_lbl = QLabel("No role stats in DB yet: " + ", ".join(unmatched))
                m_lbl.setStyleSheet("font-size: 10px; color: #8b949e;")
                m_lbl.setWordWrap(True)
                self.shortlist_lay.addWidget(m_lbl)
            self.shortlist_lay.addStretch(1)
            self.shortlist_pane.show()

            self.lbl_overall_hdr.setText(f"BEST OVERALL · TOP {len(overall)}")
            for rank, r in enumerate(overall, 1):
                self.overall_lay.addWidget(self._build_pick_card(rank, r, is_shortlist=False))
            self.overall_lay.addStretch(1)

            # Smart initial 50/50 or proportional split until the user drags the divider
            if not self._user_moved_splitter:
                avail_h = max(240, self.splitter.height() or (self.height() - 210))
                s_weight = min(max(1, len(shortlist)), 3)
                o_weight = min(max(2, len(overall)), 3)
                s_h = int(avail_h * (s_weight / (s_weight + o_weight)))
                self.splitter.setSizes([s_h, max(100, avail_h - s_h)])
        else:
            self.shortlist_pane.hide()
            self.lbl_overall_hdr.setText(f"BEST OVERALL · TOP {len(overall)}")
            for rank, r in enumerate(overall, 1):
                self.overall_lay.addWidget(self._build_pick_card(rank, r, is_shortlist=False))
            self.overall_lay.addStretch(1)

        self.shortlist_scroll.verticalScrollBar().setValue(0)
        self.overall_scroll.verticalScrollBar().setValue(0)

    # ------------------------------------------------------------------
    # Frameless Edge/Corner Resizing + Window Dragging
    # ------------------------------------------------------------------
    def _edges_at(self, pos: QPoint) -> set[str]:
        edges = set()
        if pos.x() <= RESIZE_MARGIN:
            edges.add("left")
        elif pos.x() >= self.width() - RESIZE_MARGIN:
            edges.add("right")
        if pos.y() <= RESIZE_MARGIN:
            edges.add("top")
        elif pos.y() >= self.height() - RESIZE_MARGIN:
            edges.add("bottom")
        return edges

    def _update_cursor_for_edges(self, edges: set[str]):
        if ("left" in edges and "top" in edges) or ("right" in edges and "bottom" in edges):
            self.setCursor(Qt.SizeFDiagCursor)
        elif ("right" in edges and "top" in edges) or ("left" in edges and "bottom" in edges):
            self.setCursor(Qt.SizeBDiagCursor)
        elif "left" in edges or "right" in edges:
            self.setCursor(Qt.SizeHorCursor)
        elif "top" in edges or "bottom" in edges:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def event(self, ev):
        if ev.type() == QEvent.HoverMove and not self._resize_edges:
            self._update_cursor_for_edges(self._edges_at(ev.position().toPoint()))
        return super().event(ev)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            edges = self._edges_at(event.position().toPoint())
            if edges:
                self._resize_edges = edges
                self._resize_start_global = event.globalPosition().toPoint()
                self._resize_start_geom = QRect(self.geometry())
                self._drag_pos = None
            else:
                self._resize_edges = set()
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        gpos = event.globalPosition().toPoint()
        if self._resize_edges and self._resize_start_global and self._resize_start_geom:
            dx = gpos.x() - self._resize_start_global.x()
            dy = gpos.y() - self._resize_start_global.y()
            r = QRect(self._resize_start_geom)
            min_w, min_h = self.minimumWidth(), self.minimumHeight()
            if "left" in self._resize_edges:
                r.setLeft(min(r.right() - min_w + 1, r.left() + dx))
            if "right" in self._resize_edges:
                r.setWidth(max(min_w, r.width() + dx))
            if "top" in self._resize_edges:
                r.setTop(min(r.bottom() - min_h + 1, r.top() + dy))
            if "bottom" in self._resize_edges:
                r.setHeight(max(min_h, r.height() + dy))
            self.setGeometry(r)
        elif self._drag_pos is not None:
            self.move(gpos - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._resize_edges = set()
        self._resize_start_global = None
        self._resize_start_geom = None
        self._drag_pos = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.chips_container.isVisible():
            self.chips_container.updateGeometry()
        self._size_save_timer.start()