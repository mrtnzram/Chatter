"""ChatterScreen — the top-level Kivy screen that wires all widgets together.

Layout (top to bottom)
-----------------------
1. Bird selector row      (Spinner)
2. Param row A            (MFCC Thresh | Energy Thresh | Active Region Thresh)
3. Param row B            (Min Silence | Min Bout Len | Pad)
4. Bout-select row        (BoutList | Remove Bouts | Mark as Not Outlier)
5. Bout-edit row          (Onset | Offset | Update Bout | Add Bout · Refresh | Export Bouts)
6. Status bar             (one fixed-height Label for last action feedback)
7. Zoom row               (Zoom Slider | Minor-tick TextInput)
-- divider --
9. Audio player row       (play/pause + position)
10. Scrollable spectrogram (SpectrogramView inside horizontal ScrollView)

Threading model
---------------
Slow operations (recompute, export) run on a ``threading.Thread``.
Results are marshalled back to the main thread via ``Clock.schedule_once``.
Drag interactions on the spectrogram are synchronous (no feature recompute).

Because Python threads cannot be force-killed, a stuck/hung worker is handled
by *abandoning* it: every background render captures ``self._recompute_gen``
and its result callback drops itself if the token has since advanced. The
**Refresh** button bumps that token, clears the ``_busy`` guard, and starts a
fresh load — recovering the UI from a frozen "Loading…/Computing…" state.
"""

from __future__ import annotations

import os
import sys
import threading
from functools import partial
from typing import List, Optional

import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics import Color as GColor, Rectangle as GRect
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.metrics import dp, sp
from kivy.uix.screenmanager import Screen


class _MomentumScrollEffect(DampedScrollEffect):
    """Low-friction scroll effect so flings glide instead of stopping abruptly."""
    friction = 0.01           # default is 0.05 — lower = longer, smoother glide
    min_velocity = 0.2

# Make core/ importable from this file's location
_screen_dir = os.path.dirname(__file__)
_app_dir = os.path.dirname(_screen_dir)
_core_dir = os.path.join(_app_dir, 'core')
_widgets_dir = os.path.join(_app_dir, 'widgets')
for _p in (_core_dir, _widgets_dir, _app_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chatter_controller import ChatterController # pyright: ignore
from spectrogram_view import SpectrogramView, render_spectrogram_tiles # pyright: ignore
from bout_list import BoutList # pyright: ignore
from param_input import ParamInput # pyright: ignore


# ---------------------------------------------------------------------------
# KV layout string for the static control rows
# ---------------------------------------------------------------------------

Builder.load_string("""
<_RowLabel@Label>:
    size_hint_x: None
    width: 140
    font_size: sp(12)
    color: 0.75, 0.75, 0.75, 1
    halign: 'right'
    valign: 'middle'
    text_size: self.width - 4, None

<_ActionBtn@Button>:
    size_hint_x: None
    width: 160
    size_hint_y: None
    height: 36
    font_size: sp(13)

<_FloatInput@TextInput>:
    multiline: False
    input_filter: 'float'
    size_hint_x: None
    width: 90
    size_hint_y: None
    height: 34
    font_size: sp(13)
    background_color: 0.12, 0.12, 0.12, 1
    foreground_color: 1, 1, 1, 1
""")


# ---------------------------------------------------------------------------
# ChatterScreen
# ---------------------------------------------------------------------------

class ChatterScreen(Screen):

    def __init__(self, controller: ChatterController,
                 bouts_audio_dir: str = 'bouts_audio',
                 on_back=None, **kwargs):
        super().__init__(**kwargs)
        self.ctrl = controller
        self._bouts_audio_dir = bouts_audio_dir
        self._on_back_cb = on_back
        self._busy = False       # True while a background thread is running
        self._current_idx: int = 0
        self._zoom_debounce: Optional[object] = None
        self._status_clear_event: Optional[object] = None
        # Monotonic token identifying the latest background render. A worker's
        # result callback ignores itself if this has advanced since it started
        # (used to abandon a hung/superseded computation — see Refresh).
        self._recompute_gen: int = 0

        self._build_ui()
        self._connect_callbacks()

        # Populate bird list and trigger initial load
        options = self.ctrl.get_bird_options()
        self._bird_spinner.values = [label for label, _, _ in options]
        self._bird_index_map = {label: idx for label, idx, _ in options}
        if options:
            self._bird_spinner.text = options[0][0]
            # Initial load happens via on_text binding

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = BoxLayout(orientation='vertical', padding=18, spacing=14)
        self.add_widget(root)

        # ---- Control panel (auto-sizes to its content so nothing clips) ----
        ctrl_panel = BoxLayout(orientation='vertical', size_hint_y=None,
                               spacing=12)
        ctrl_panel.bind(minimum_height=ctrl_panel.setter('height'))
        root.add_widget(ctrl_panel)

        # Row 1: Bird selector (its own row, visually separated below)
        row1 = BoxLayout(size_hint_y=None, height=56, spacing=12)
        _lbl(row1, 'Select Bird:', font_size=15)
        self._bird_spinner = Spinner(
            text='', font_size=sp(18),
            size_hint_x=1, size_hint_y=1,
            background_color=(0.18, 0.40, 0.62, 1),
            background_normal='',
            color=(1, 1, 1, 1),
            # Bounded, scrollable dropdown list
            dropdown_cls=partial(DropDown, max_height=360),
        )
        self._prev_btn = _btn(row1, '< Prev',
                              bg=(0.18, 0.35, 0.55, 1), width=90, height=56,
                              font_size=15)
        row1.add_widget(self._bird_spinner)
        self._next_btn = _btn(row1, 'Next >',
                              bg=(0.18, 0.35, 0.55, 1), width=90, height=56,
                              font_size=15)
        self._exported_label = Label(
            text='', color=(0.25, 0.85, 0.45, 1),
            size_hint_x=None, width=dp(160),
            font_size=sp(14), halign='center', valign='middle',
        )
        self._exported_label.bind(size=self._exported_label.setter('text_size'))
        row1.add_widget(self._exported_label)
        self._back_btn = _btn(row1, 'New Project',
                              bg=(0.20, 0.20, 0.25, 1), width=150, height=56,
                              font_size=17)
        ctrl_panel.add_widget(row1)

        # Divider separating the bird selector from the parameter rows
        _divider(ctrl_panel)

        # Row 2: Detection params A — centred in the available width
        row2_wrap = AnchorLayout(size_hint_y=None, height=50, anchor_x='center')
        row2 = BoxLayout(size_hint_x=0.75, spacing=20)
        self._mfcc_thresh = _param_input(row2, 'MFCC Thresh:', 0.5)
        self._energy_thresh = _param_input(row2, 'Energy Thresh:', 0.1)
        self._active_thresh = _param_input(row2, 'Active Region Thresh:', 0.001)
        row2_wrap.add_widget(row2)
        ctrl_panel.add_widget(row2_wrap)

        # Row 3: Detection params B — centred in the available width
        row3_wrap = AnchorLayout(size_hint_y=None, height=50, anchor_x='center')
        row3 = BoxLayout(size_hint_x=0.75, spacing=20)
        self._min_silence = _param_input(row3, 'Min Silence:', 0.9)
        self._min_bout_len = _param_input(row3, 'Min Bout Len:', 1.0)
        self._pad = _param_input(row3, 'Pad:', 0.5)
        row3_wrap.add_widget(row3)
        ctrl_panel.add_widget(row3_wrap)

        _divider(ctrl_panel)

        # Row 4: Bout list (tall — shows several bouts) + Remove + Not Outlier
        row4 = BoxLayout(size_hint_y=None, height=190, spacing=14)
        self._bout_list = BoutList(size_hint_x=1, size_hint_y=1)
        row4.add_widget(self._bout_list)
        btn_col = BoxLayout(orientation='vertical', size_hint_x=None,
                            width=dp(200), spacing=12)
        self._remove_btn = _btn(btn_col, 'Remove Bouts',
                                bg=(0.65, 0.15, 0.15, 1), height=52,
                                width=200, font_size=15)
        self._not_outlier_btn = _btn(btn_col, 'Mark Not Outlier',
                                     bg=(0.15, 0.35, 0.65, 1), height=52,
                                     width=200, font_size=15)
        row4.add_widget(btn_col)
        ctrl_panel.add_widget(row4)

        # Row 5: Onset / Offset / Update / Add (left) · Refresh / Export (right).
        # A flexible spacer pushes Refresh + Export to the right edge so they
        # sit on the same baseline as the onset/offset inputs.
        row5 = BoxLayout(size_hint_y=None, height=56, spacing=12)
        _lbl(row5, 'Onset:')
        self._onset_input = _float_input(row5, '0.000', font_size=20, width=130)
        _lbl(row5, 'Offset:')
        self._offset_input = _float_input(row5, '0.000', font_size=20, width=130)
        self._update_btn = _btn(row5, 'Update Bout',
                                bg=(0.65, 0.45, 0.0, 1),
                                font_size=22, height=56, width=210)
        self._add_btn = _btn(row5, 'Add Bout',
                             bg=(0.1, 0.5, 0.1, 1),
                             font_size=22, height=56, width=210)
        row5.add_widget(Widget(size_hint_x=1))   # flexible spacer
        self._refresh_btn = _btn(row5, 'Refresh',
                                 bg=(0.0, 0.55, 0.55, 1), height=56,
                                 width=200, font_size=16)
        self._export_btn = _btn(row5, 'Export Bouts',
                                bg=(0.0, 0.35, 0.6, 1), height=56,
                                width=200, font_size=16)
        ctrl_panel.add_widget(row5)

        # Status bar — fixed height (reserves room for up to two lines) so the
        # spectrogram below never shifts as messages appear/clear/wrap.
        self._status_label = Label(
            text='Ready.',
            size_hint_y=None, height=dp(52),
            font_size=sp(20), color=(0.7, 0.9, 0.7, 1),
            halign='left', valign='top',
        )
        self._status_label.bind(
            width=lambda inst, w: setattr(inst, 'text_size', (w, None))
        )
        ctrl_panel.add_widget(self._status_label)

        # Row 7: Zoom + minor tick
        row7 = BoxLayout(size_hint_y=None, height=48, spacing=14)
        _lbl(row7, 'Zoom:', font_size=16)
        self._zoom_slider = Slider(
            min=0.5, max=3.0, value=1.0, step=0.1,
            size_hint_x=1, size_hint_y=1,
        )
        row7.add_widget(self._zoom_slider)
        _lbl(row7, 'Minor tick (s):', font_size=16, width=180)
        self._minor_tick_input = _float_input(row7, '0.1')
        ctrl_panel.add_widget(row7)

        # ---- Spectrogram area ----------------------------------------
        spec_area = BoxLayout(orientation='vertical', spacing=10)
        root.add_widget(spec_area)

        # Top bar: shift-to-add hint + loading indicator.
        # text_size is bound to width-only (not height) so that font_size
        # changes actually take effect — binding to self.size clips text to the
        # fixed box height, making font changes invisible.
        top_bar = BoxLayout(size_hint_y=None, spacing=12)
        top_bar.bind(minimum_height=top_bar.setter('height'))
        hint = Label(
            text='Drag to scroll  ·  Shift + Drag to add a bout  ·  Cmd/Ctrl + Drag a boundary line to edit it  ·  left / right to switch recordings · up / down to navigate bouts',
            size_hint_x=1, size_hint_y=None,
            font_size=sp(15), color=(0.6, 0.6, 0.65, 1),
            halign='left', valign='top',
        )
        hint.bind(width=lambda inst, w: setattr(inst, 'text_size', (w, None)))
        hint.bind(texture_size=lambda inst, ts: setattr(inst, 'height', ts[1] + 10))
        top_bar.add_widget(hint)
        self._loading_label = Label(
            text='', size_hint_x=None, width=dp(200),
            font_size=sp(18), color=(1, 0.75, 0.0, 1),
        )
        top_bar.add_widget(self._loading_label)
        spec_area.add_widget(top_bar)

        # Scrollable spectrogram.
        # scroll_type=['bars', 'content'] enables both the scrollbar and
        # content-drag/fling scrolling.  A low-friction momentum effect makes
        # flings glide.  A plain drag scrolls; Shift+drag is captured by the
        # SpectrogramView to draw a new bout; dragging a bout's onset/offset
        # line always captures.
        self._scroll = ScrollView(
            do_scroll_x=True, do_scroll_y=False,
            size_hint=(1, 1),
            scroll_type=['bars', 'content'],
            scroll_wheel_distance=80,
            effect_cls=_MomentumScrollEffect,
            bar_width=16,
            bar_color=(0.4, 0.6, 1.0, 0.9),
            bar_inactive_color=(0.3, 0.3, 0.3, 0.6),
        )
        self._spec_view = SpectrogramView(size_hint=(None, 1))
        self._scroll.add_widget(self._spec_view)
        spec_area.add_widget(self._scroll)

    # ------------------------------------------------------------------
    # Callback wiring
    # ------------------------------------------------------------------

    def _connect_callbacks(self):
        # Bird selection
        self._bird_spinner.bind(text=self._on_bird_selected)

        # Param commits → force re-detection with new params. A forced
        # recompute re-runs detection and updates the in-memory bouts list, so
        # parameters are effectively finalized on every change (Finalize removed).
        for pi in (self._mfcc_thresh, self._energy_thresh, self._active_thresh,
                   self._min_silence, self._min_bout_len):
            pi.on_commit = lambda _: self._schedule_recompute(force=True)
        # Pad only affects clip boundaries (wavstart/wavend), not detection
        self._pad.on_commit = lambda _: self._schedule_recompute(force=False)

        # Zoom / minor-tick → base re-render only (no feature recompute)
        self._zoom_slider.bind(value=self._on_zoom_changed)
        self._minor_tick_input.bind(on_text_validate=lambda inst: self._on_minor_tick_commit())
        self._minor_tick_input.bind(focus=lambda inst, foc: (
            self._on_minor_tick_commit() if not foc else None
        ))

        # Bout list selection → populate onset/offset boxes + spec lines
        self._bout_list.on_selection = self._on_bout_selection_changed

        # Bout edit buttons
        self._update_btn.bind(on_release=self._on_update_bout)
        self._add_btn.bind(on_release=self._on_add_bout)
        self._remove_btn.bind(on_release=self._on_remove_bouts)
        self._not_outlier_btn.bind(on_release=self._on_mark_not_outlier)

        # Prev / Next bird navigation
        self._prev_btn.bind(on_release=self._on_prev_bird)
        self._next_btn.bind(on_release=self._on_next_bird)

        # Action buttons
        self._back_btn.bind(on_release=self._on_back)
        self._refresh_btn.bind(on_release=self._on_refresh)
        self._export_btn.bind(on_release=self._on_export)

        # Spectrogram interactive callbacks
        self._spec_view.on_onset_live = self._on_onset_live
        self._spec_view.on_offset_live = self._on_offset_live
        self._spec_view.on_bout_updated = self._on_drag_commit_update
        self._spec_view.on_bout_added = self._on_drag_commit_add
        self._spec_view.on_bout_selected = self._on_spec_bout_selected

        # Arrow-key bout navigation
        Window.bind(on_key_down=self._on_key_down)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def _on_key_down(self, window, key, scancode, codepoint, modifier):
        # Don't hijack arrow keys while the user is typing in a text field
        # (parameter values, onset/offset) — let the field handle them.
        if self._text_input_focused():
            return False

        # Left/right arrows → previous/next recording (work even with no bouts)
        if key == 276:   # left arrow → previous recording
            self._on_prev_bird()
            return True
        if key == 275:   # right arrow → next recording
            self._on_next_bird()
            return True

        bouts = self.ctrl.current_bouts.get(self._current_idx, [])
        if not bouts:
            return False
        n = len(bouts)
        sel = self._bout_list.selected_ids

        if key == 274:   # down arrow → next bout
            new_id = (sel[0] + 1) if sel else 0
            new_id = min(new_id, n - 1)
        elif key == 273:  # up arrow → previous bout
            new_id = (sel[0] - 1) if sel else n - 1
            new_id = max(new_id, 0)
        else:
            return False

        self._bout_list.set_selection([new_id])
        self._scroll_spec_to_bout(new_id)
        return True

    def _text_input_focused(self) -> bool:
        """True if any TextInput within this screen currently has focus."""
        for w in self.walk(restrict=True):
            if isinstance(w, TextInput) and w.focus:
                return True
        return False

    def _scroll_spec_to_bout(self, bout_id: int):
        """Scroll the spectrogram so the selected bout is centred in the viewport."""
        bouts = self.ctrl.current_bouts.get(self._current_idx, [])
        if bout_id >= len(bouts):
            return
        g = self._spec_view._geometry
        if g is None:
            return
        bout = bouts[bout_id]
        center_px = (g.time_to_x(bout['onset']) + g.time_to_x(bout['offset'])) / 2.0

        content_w = self._spec_view.width
        viewport_w = self._scroll.width
        if content_w <= viewport_w:
            return

        target = (center_px - viewport_w / 2.0) / (content_w - viewport_w)
        self._scroll.scroll_x = float(max(0.0, min(1.0, target)))

    # ------------------------------------------------------------------
    # Screen lifecycle
    # ------------------------------------------------------------------

    def on_enter(self, *_):
        """Called by Kivy when the screen transition into this screen completes.

        The FadeTransition runs for 0.25 s.  If the background recompute
        finishes during that window, set_base_tiles() draws into a canvas
        that Kivy may not re-composite after the transition ends.  Forcing
        a full redraw here guarantees the spectrogram is visible as soon as
        the screen settles.
        """
        if self._spec_view._textures:
            self._spec_view._full_redraw()

    # ------------------------------------------------------------------
    # Bird selection
    # ------------------------------------------------------------------

    def _on_prev_bird(self, *_):
        values = self._bird_spinner.values
        if not values:
            return
        try:
            pos = list(values).index(self._bird_spinner.text)
        except ValueError:
            pos = 0
        self._bird_spinner.text = values[(pos - 1) % len(values)]

    def _on_next_bird(self, *_):
        values = self._bird_spinner.values
        if not values:
            return
        try:
            pos = list(values).index(self._bird_spinner.text)
        except ValueError:
            pos = 0
        self._bird_spinner.text = values[(pos + 1) % len(values)]

    def _on_bird_selected(self, spinner, text):
        if not text or text not in self._bird_index_map:
            return
        idx = self._bird_index_map[text]
        self._current_idx = idx
        # Restore saved params for this bird
        params = self.ctrl.get_params(idx)
        self._set_param_widgets(params)
        # Clear spectrogram while loading
        self._spec_view.clear()
        self._bout_list.set_bouts([])
        self._set_status('Loading...')
        self._update_exported_label()
        self._schedule_recompute()

    # ------------------------------------------------------------------
    # Recompute pipeline (threaded)
    # ------------------------------------------------------------------

    def _schedule_recompute(self, force: bool = False):
        if self._busy:
            return
        self._busy = True
        self._recompute_gen += 1
        gen = self._recompute_gen
        self._loading_label.text = 'Computing...'
        self._set_status('Running feature detection...')
        params = self._read_params()
        idx = self._current_idx

        def _worker():
            try:
                bouts, features = self.ctrl.recompute(idx, params, force=force)
                # Also prepare the spectrogram texture off-thread
                row = self.ctrl.df.iloc[idx]
                chunk_start = float(row.get('chunk_start') or 0.0)
                S_db, sr = self.ctrl.get_cached_spectrogram(
                    row['wav_location'], row['audio'], row['sr'],
                    chunk_start=chunk_start,
                )
                zoom = self._zoom_slider.value
                minor = self._read_minor_tick()
                tiles, geometry = render_spectrogram_tiles(
                    S_db, int(sr), self.ctrl.extractor.hop_length,
                    zoom_factor=zoom, minor_tick_step=minor,
                )
                Clock.schedule_once(
                    partial(self._on_recompute_done, gen, idx, bouts, tiles,
                            geometry, row['wav_location']),
                    0,
                )
            except Exception as exc:
                Clock.schedule_once(
                    partial(self._on_recompute_error, gen, str(exc)), 0
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_recompute_done(self, gen, idx, bouts, tiles, geometry, wav_path, dt):
        if gen != self._recompute_gen:
            # A newer load (Refresh, param change, or bird switch) superseded
            # this one. Drop the stale result; the newer worker owns _busy.
            return
        if idx != self._current_idx:
            # User switched birds mid-load. Release the guard and load the bird
            # they actually want now.
            self._set_busy(False)
            self._schedule_recompute()
            return
        self._spec_view.set_base_tiles(tiles, geometry)
        self._spec_view.update_bouts(bouts, [])
        self._bout_list.set_bouts(bouts)
        self._loading_label.text = ''
        self._set_status(f'Loaded {len(bouts)} bout(s).')
        self._set_busy(False)

    def _on_recompute_error(self, gen, msg, dt):
        if gen != self._recompute_gen:
            return
        self._set_status(
            'Could not load recording. Check that the WAV file exists, '
            'or try adjusting the zoom.'
        )
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Refresh — safety hatch for a frozen / stuck load
    # ------------------------------------------------------------------

    def _on_refresh(self, *_):
        """Force a spectrogram reload, recovering from a stuck 'busy' state.

        Python threads can't be force-killed, so any in-flight worker is
        *abandoned* rather than stopped: bumping ``_recompute_gen`` makes its
        eventual result a no-op, and clearing ``_busy`` lets a fresh load
        start immediately. If the previous computation had actually finished
        (the common "computed but never displayed" freeze), the spectrogram is
        cached and this returns near-instantly.
        """
        self._recompute_gen += 1   # invalidate any in-flight worker's result
        self._busy = False
        self._loading_label.text = ''
        self._set_status('Refreshing spectrogram...')
        self._schedule_recompute(force=False)

    # ------------------------------------------------------------------
    # Base re-render (zoom/minor-tick change — no feature recompute)
    # ------------------------------------------------------------------

    def _on_zoom_changed(self, slider, value):
        if self._zoom_debounce:
            self._zoom_debounce.cancel()
        self._zoom_debounce = Clock.schedule_once(
            lambda _: self._redraw_base(), 0.25
        )

    def _on_minor_tick_commit(self, *_):
        self._redraw_base()

    def _redraw_base(self):
        if self._busy:
            return
        idx = self._current_idx
        row = self.ctrl.df.iloc[idx]
        if not isinstance(row.get('audio'), np.ndarray):
            return
        self._busy = True
        self._recompute_gen += 1
        gen = self._recompute_gen
        self._loading_label.text = 'Rendering...'

        chunk_start = float(row.get('chunk_start') or 0.0)

        def _worker():
            try:
                S_db, sr = self.ctrl.get_cached_spectrogram(
                    row['wav_location'], row['audio'], row['sr'],
                    chunk_start=chunk_start,
                )
                zoom = self._zoom_slider.value
                minor = self._read_minor_tick()
                tiles, geometry = render_spectrogram_tiles(
                    S_db, int(sr), self.ctrl.extractor.hop_length,
                    zoom_factor=zoom, minor_tick_step=minor,
                )
                bouts = self.ctrl.current_bouts.get(idx, [])
                sel = self._bout_list.selected_ids
                Clock.schedule_once(
                    partial(self._on_base_done, gen, tiles, geometry, bouts, sel), 0
                )
            except Exception:
                Clock.schedule_once(partial(self._on_base_error, gen), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_base_done(self, gen, tiles, geometry, bouts, sel, dt):
        if gen != self._recompute_gen:
            return
        self._spec_view.set_base_tiles(tiles, geometry)
        self._spec_view.update_bouts(bouts, sel)
        self._loading_label.text = ''
        self._set_status('')   # dismiss any lingering render-error message
        self._set_busy(False)

    def _on_base_error(self, gen, dt):
        if gen != self._recompute_gen:
            return
        self._set_status(
            'Could not render spectrogram. '
            'Try zooming out or increasing the minor tick step.'
        )
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Bout list selection
    # ------------------------------------------------------------------

    def _on_bout_selection_changed(self, selected_ids: list):
        bouts = self.ctrl.current_bouts.get(self._current_idx, [])
        if len(selected_ids) == 1:
            bid = selected_ids[0]
            if bid < len(bouts):
                self._onset_input.text = f'{bouts[bid]["onset"]:.3f}'
                self._offset_input.text = f'{bouts[bid]["offset"]:.3f}'
        self._spec_view.update_bouts(bouts, selected_ids)

    def _on_spec_bout_selected(self, bid: int, extend: bool):
        """Fired when the user clicks a bout span in the spectrogram."""
        self._bout_list.select_row(bid, extend=extend)

    # ------------------------------------------------------------------
    # Drag live-update (onset/offset boxes stay in sync during drag)
    # ------------------------------------------------------------------

    def _on_onset_live(self, t: float):
        self._onset_input.text = f'{t:.3f}'

    def _on_offset_live(self, t: float):
        self._offset_input.text = f'{t:.3f}'

    # ------------------------------------------------------------------
    # Drag commit — Update Bout (from spectrogram drag)
    # ------------------------------------------------------------------

    def _on_drag_commit_update(self, bout_id: int, onset: float, offset: float):
        result = self.ctrl.update_bout(self._current_idx, bout_id, onset, offset)
        ok, msg = result[0], result[1]
        new_id = result[2] if len(result) > 2 else bout_id
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation(sel=[new_id])
        else:
            bouts = self.ctrl.current_bouts.get(self._current_idx, [])
            self._spec_view.update_bouts(bouts, self._bout_list.selected_ids)

    def _on_drag_commit_add(self, onset: float, offset: float):
        ok, msg = self.ctrl.add_bout(self._current_idx, onset, offset)
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_update_bout(self, *_):
        selected = self._bout_list.selected_ids
        if len(selected) != 1:
            self._set_status('Select exactly one bout to update.')
            return
        try:
            onset = float(self._onset_input.text)
            offset = float(self._offset_input.text)
        except ValueError:
            self._set_status('Invalid onset/offset values.')
            return
        result = self.ctrl.update_bout(self._current_idx, selected[0], onset, offset)
        ok, msg = result[0], result[1]
        new_id = result[2] if len(result) > 2 else selected[0]
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation(sel=[new_id])

    def _on_add_bout(self, *_):
        try:
            onset = float(self._onset_input.text)
            offset = float(self._offset_input.text)
        except ValueError:
            self._set_status('Invalid onset/offset values.')
            return
        ok, msg = self.ctrl.add_bout(self._current_idx, onset, offset)
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation()

    def _on_remove_bouts(self, *_):
        selected = self._bout_list.selected_ids
        if not selected:
            self._set_status('No bouts selected.')
            return
        ok, msg = self.ctrl.remove_bouts(self._current_idx, selected)
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation()

    def _on_mark_not_outlier(self, *_):
        selected = self._bout_list.selected_ids
        if not selected:
            self._set_status('No bouts selected.')
            return
        ok, msg = self.ctrl.set_not_outlier(self._current_idx, selected)
        self._set_status(msg)
        if ok:
            self._refresh_bouts_after_mutation()

    def _on_back(self, *_):
        """Confirm, then hand off to the app to tear down and return to welcome."""
        if not self._on_back_cb:
            return

        content = BoxLayout(orientation='vertical', spacing=14,
                            padding=(20, 14))
        content.add_widget(Label(
            text='Return to the welcome screen?\nAny un-exported bout edits will be lost.',
            halign='center', valign='middle',
            font_size=sp(16), color=(0.85, 0.85, 0.88, 1),
        ))
        btn_row = BoxLayout(size_hint_y=None, height=44, spacing=12)
        cancel_btn = Button(
            text='Cancel',
            background_color=(0.22, 0.22, 0.26, 1), background_normal='',
        )
        confirm_btn = Button(
            text='Yes, new project',
            background_color=(0.55, 0.18, 0.18, 1), background_normal='',
        )
        btn_row.add_widget(cancel_btn)
        btn_row.add_widget(confirm_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title='New Project',
            content=content,
            size_hint=(None, None), size=(dp(400), dp(180)),
            auto_dismiss=False,
        )
        cancel_btn.bind(on_release=popup.dismiss)

        def _confirm(*_):
            popup.dismiss()
            self._on_back_cb()

        confirm_btn.bind(on_release=_confirm)
        popup.open()

    def _on_export(self, *_):
        if self._busy:
            return
        self._busy = True
        self._loading_label.text = 'Exporting...'
        idx = self._current_idx

        def _worker():
            ok, msg = self.ctrl.export(idx, output_dir=self._bouts_audio_dir)
            Clock.schedule_once(partial(self._on_export_done, msg), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_export_done(self, msg, dt):
        self._loading_label.text = ''
        self._set_status(msg)
        self._set_busy(False)
        self._update_exported_label()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_exported_label(self):
        if self.ctrl.is_exported(self._current_idx):
            self._exported_label.text = 'Already exported'
        else:
            self._exported_label.text = ''

    def _refresh_bouts_after_mutation(self, sel: list | None = None):
        """Sync bout list + spectrogram overlay after any in-memory mutation.

        ``sel`` — explicit selection to restore (use when the caller knows the
        correct post-sort index, e.g. after update_bout).  Falls back to the
        current list selection when omitted.
        """
        bouts = self.ctrl.current_bouts.get(self._current_idx, [])
        if sel is None:
            sel = [i for i in self._bout_list.selected_ids if i < len(bouts)]
        else:
            sel = [i for i in sel if i < len(bouts)]
        self._bout_list.set_bouts(bouts)
        if sel:
            self._bout_list.set_selection(sel)
        self._spec_view.update_bouts(bouts, sel)

    def _read_params(self) -> dict:
        return {
            'mfcc_threshold': self._mfcc_thresh.value,
            'energy_threshold': self._energy_thresh.value,
            'active_region_thresh': self._active_thresh.value,
            'min_silence': self._min_silence.value,
            'min_bout_len': self._min_bout_len.value,
            'pad': self._pad.value,
        }

    def _set_param_widgets(self, params: dict):
        self._mfcc_thresh.value = params.get('mfcc_threshold', 0.5)
        self._energy_thresh.value = params.get('energy_threshold', 0.1)
        self._active_thresh.value = params.get('active_region_thresh', 0.001)
        self._min_silence.value = params.get('min_silence', 0.9)
        self._min_bout_len.value = params.get('min_bout_len', 1.0)
        self._pad.value = params.get('pad', 0.5)

    def _read_minor_tick(self) -> float:
        try:
            return float(self._minor_tick_input.text)
        except ValueError:
            return 0.1

    _STATUS_CLEAR_DELAY = 5.0  # seconds before a status message auto-clears

    def _set_status(self, msg, dt=None):
        self._status_label.text = str(msg)
        if self._status_clear_event:
            self._status_clear_event.cancel()
            self._status_clear_event = None
        if msg:
            self._status_clear_event = Clock.schedule_once(
                lambda *_: setattr(self._status_label, 'text', ''),
                self._STATUS_CLEAR_DELAY,
            )

    def _set_busy(self, busy: bool):
        self._busy = busy
        if not busy:
            self._loading_label.text = ''


# ---------------------------------------------------------------------------
# Helper factory functions (keeps _build_ui readable)
# ---------------------------------------------------------------------------

def _divider(parent, height: int = 2):
    """A thin horizontal separator line between control rows."""
    d = Widget(size_hint_y=None, height=height)
    with d.canvas:
        GColor(0.32, 0.32, 0.38, 1)
        rect = GRect(pos=d.pos, size=d.size)
    d.bind(pos=lambda *a: setattr(rect, 'pos', d.pos),
           size=lambda *a: setattr(rect, 'size', d.size))
    parent.add_widget(d)
    return d


def _lbl(parent, text: str, font_size: int = 16, width: int = 140) -> Label:
    # size_hint_y=1 → fill the row height so valign='middle' truly centres the
    # text against the inputs/buttons beside it (BoxLayout ignores pos_hint).
    lbl = Label(
        text=text,
        size_hint_x=None, width=dp(width),
        size_hint_y=1,
        font_size=sp(font_size),
        color=(0.8, 0.8, 0.8, 1),
        halign='right', valign='middle',
    )
    lbl.bind(size=lbl.setter('text_size'))
    parent.add_widget(lbl)
    return lbl


def _param_input(parent, label: str, default: float) -> ParamInput:
    pi = ParamInput(default=default, label=label, size_hint_x=1)
    parent.add_widget(pi)
    return pi


def _float_input(parent, default_text: str = '0.0', font_size: int = 15,
                 width: int = 110, height: int = 44) -> TextInput:
    # Fill the row vertically and centre the single line of text so the box
    # lines up with its label (a single-line TextInput top-aligns by default).
    ti = TextInput(
        text=default_text,
        multiline=False,
        input_filter='float',
        size_hint_x=None, width=dp(width),
        size_hint_y=1,
        font_size=sp(font_size),
        background_color=(0.12, 0.12, 0.12, 1),
        foreground_color=(1, 1, 1, 1),
        cursor_color=(1, 1, 1, 1),
    )

    def _center(inst, *_a):
        inst.padding = [8, max((inst.height - inst.line_height) / 2.0, 0), 8, 0]

    ti.bind(height=_center, line_height=_center)
    Clock.schedule_once(lambda _dt: _center(ti), 0)
    parent.add_widget(ti)
    return ti


def _btn(parent, text: str, bg=(0.2, 0.2, 0.2, 1), font_size: int = 19,
         width: int = 190, height: int = 48) -> Button:
    b = Button(
        text=text,
        size_hint_x=None, width=dp(width),
        size_hint_y=None, height=height,
        font_size=sp(font_size),
        background_color=bg,
        background_normal='',
    )
    parent.add_widget(b)
    return b
