"""SpectrogramView — matplotlib base tiles rendered as Kivy textures, with a
Kivy Graphics overlay for bout spans, draggable lines, the draft-add rectangle,
and the audio playhead.

Architecture
------------
- The spectrogram *base* (STFT → dB → grey colormap) is rendered off-thread
  into one or more RGBA numpy arrays by ``render_spectrogram_tiles()``.
- Each chunk is uploaded to a ``kivy.graphics.texture.Texture`` and drawn as a
  ``Rectangle`` in ``canvas``.
- All interactive elements (bout spans, onset/offset lines, playhead, draft
  region) are pure Kivy ``Graphics`` instructions in ``canvas.after``; they
  never re-render matplotlib and update synchronously on every drag step.

Coordinate system
-----------------
The widget lives inside a horizontal ``ScrollView`` at its natural pixel size
(``size = (total_w_px, height_px)``), so widget-local x == texture pixel x
for 1:1 display.  ``SpectrogramGeometry.time_to_x / x_to_time`` are the single
source of truth for the px↔time mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget

# Maximum texture dimension safe on most desktop GPUs.
_MAX_TILE_PX = 8192
_SPEC_HEIGHT_PX = 400  # base spectrogram tile height in pixels
_SPEC_DPI = 100        # DPI used for the matplotlib render
_GRAB_PX = 24          # touch hit-test radius around a line
_SELECT_DRAG_PX = 10   # horizontal movement threshold that cancels a select-tap
_AXIS_H = 40           # height (px) of the top time-axis ruler strip
_TICK_FONT = 22        # font size for time-axis labels
_BOUT_FONT = 30        # font size for bout number labels


# ---------------------------------------------------------------------------
# SpectrogramGeometry — shared px↔time utility
# ---------------------------------------------------------------------------

@dataclass
class SpectrogramGeometry:
    """Describes the rendered spectrogram canvas dimensions and time range."""
    total_w_px: int
    height_px: int
    duration: float         # seconds
    minor_tick_step: float = 0.1   # minor tick spacing (s) for the ruler

    def time_to_x(self, t: float) -> float:
        """Seconds → widget-local x pixel (float)."""
        if self.duration <= 0:
            return 0.0
        return (t / self.duration) * self.total_w_px

    def x_to_time(self, x: float) -> float:
        """Widget-local x pixel → seconds, clamped to [0, duration]."""
        if self.total_w_px <= 0:
            return 0.0
        t = x / self.total_w_px * self.duration
        return max(0.0, min(self.duration, t))


# ---------------------------------------------------------------------------
# Off-thread tile renderer (call from threading.Thread)
# ---------------------------------------------------------------------------

def render_spectrogram_tiles(
    S_db: np.ndarray,
    sr: int,
    hop_length: int,
    zoom_factor: float = 1.0,
    minor_tick_step: float = 0.1,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> Tuple[List[Tuple[np.ndarray, int, int]], SpectrogramGeometry]:
    """Render the spectrogram as a list of RGBA tile arrays.

    Parameters
    ----------
    S_db:
        dB-scaled STFT magnitude, shape ``(n_freqs, n_frames)``.
    sr, hop_length:
        Used to compute total audio duration.
    zoom_factor:
        Horizontal zoom multiplier (matches the notebook's zoom slider).
    minor_tick_step:
        Minor tick spacing in seconds (carried through for tick drawing in the
        overlay; this function does not draw ticks).

    Returns
    -------
    tiles:
        ``[(rgba_array, x_start_px, tile_w_px), ...]`` where ``rgba_array``
        has dtype ``uint8`` and shape ``(height_px, tile_w_px, 4)``.
    geometry:
        ``SpectrogramGeometry`` instance for px↔time conversion.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.cm as cm

    n_frames = S_db.shape[1]
    n_freqs = S_db.shape[0]
    duration = n_frames * hop_length / sr

    # Match the notebook's width formula
    width_inch = max(duration / (5.0 / 7.0) * zoom_factor, 15.0)
    total_w_px = int(width_inch * _SPEC_DPI)
    height_px = _SPEC_HEIGHT_PX

    # Normalise to [0, 1] for the grey colormap
    s_min, s_max = S_db.min(), S_db.max()
    if s_max > s_min:
        S_norm = (S_db - s_min) / (s_max - s_min)
    else:
        S_norm = np.zeros_like(S_db)
    # Display-only brightness/contrast tint: pivot contrast about mid-grey,
    # then shift by brightness. Pure pixel remap — S_db (and its cache) untouched.
    if contrast != 1.0 or brightness != 0.0:
        S_norm = np.clip((S_norm - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    # Flip vertically: low frequencies at bottom
    S_norm_flip = S_norm[::-1, :]

    n_tiles = math.ceil(total_w_px / _MAX_TILE_PX)
    tiles: List[Tuple[np.ndarray, int, int]] = []

    for tile_i in range(n_tiles):
        x_start = tile_i * _MAX_TILE_PX
        x_end = min(x_start + _MAX_TILE_PX, total_w_px)
        tile_w = x_end - x_start

        # Frame range for this tile
        f_start = int(x_start / total_w_px * n_frames)
        f_end = int(x_end / total_w_px * n_frames)
        f_end = min(f_end, n_frames)

        S_tile = S_norm_flip[:, f_start:f_end]

        # Resize to exact pixel dimensions with matplotlib imshow (handles
        # arbitrary aspect ratios without scipy dependency)
        fig_w_inch = tile_w / _SPEC_DPI
        fig_h_inch = height_px / _SPEC_DPI
        fig, ax = plt.subplots(figsize=(fig_w_inch, fig_h_inch), dpi=_SPEC_DPI)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(
            S_tile,
            aspect='auto',
            origin='upper',   # already flipped
            cmap='gray',
            interpolation='nearest',
            vmin=0.0, vmax=1.0,
        )
        ax.axis('off')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(height_px, tile_w, 4).copy()
        plt.close(fig)

        tiles.append((rgba, x_start, tile_w))

    geometry = SpectrogramGeometry(
        total_w_px=total_w_px,
        height_px=height_px,
        duration=duration,
        minor_tick_step=minor_tick_step,
    )
    return tiles, geometry


def _nice_step(target: float) -> float:
    """Round ``target`` up to the nearest 1/2/5 × 10ⁿfff 'nice' number."""
    if target <= 0:
        return 1.0
    exp = math.floor(math.log10(target))
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        step = mult * base
        if step >= target:
            return step
    return 10 * base


# ---------------------------------------------------------------------------
# SpectrogramView widget
# ---------------------------------------------------------------------------

class SpectrogramView(Widget):
    """Interactive spectrogram canvas.

    Public API (called by ChatterScreen)
    =====================================
    set_base_tiles(tiles, geometry)
        Upload rendered tiles to GPU textures and redraw.  Must be called on
        the main thread (via ``Clock.schedule_once``).
    update_bouts(bouts, selected_ids)
        Refresh overlay without re-rendering the base.
    set_playhead(t)
        Move the playhead line to time ``t``.
    clear()
        Remove base and overlay (e.g. while a new bird is loading).

    Callbacks (set by ChatterScreen before use)
    ============================================
    on_onset_live(t)     — fired every drag step for the onset line
    on_offset_live(t)    — fired every drag step for the offset line
    on_bout_updated(bout_id, onset, offset)  — drag released on existing line
    on_bout_added(onset, offset)             — add-drag released
    on_seek(t)                               — tiny click (no drag intent)
    """

    # Colour palette matching plot_bout_overlays conventions
    _COL_NORMAL_SPAN  = (0.0, 0.8, 0.0, 0.15)
    _COL_OUTLIER_SPAN = (1.0, 0.0, 0.0, 0.30)
    _COL_SELECTED_SPAN= (1.0, 0.85, 0.0, 0.35)  # amber — "in progress"
    _COL_ONSET_NORMAL = (0.0, 0.8, 0.0, 1.0)
    _COL_ONSET_OUTLIER= (1.0, 0.0, 0.0, 1.0)
    _COL_OFFSET       = (0.2, 0.5, 1.0, 1.0)    # blue
    _COL_DRAFT        = (1.0, 0.55, 0.0, 1.0)   # amber
    _COL_PLAYHEAD     = (1.0, 1.0, 0.0, 0.9)    # yellow

    def __init__(self, **kwargs):
        # Default size_hint_x=None so the widget can be wider than the
        # ScrollView. size_hint_y=1 lets it fill the scroll area vertically.
        kwargs.setdefault('size_hint_x', None)
        kwargs.setdefault('size_hint_y', 1)
        super().__init__(**kwargs)
        self._textures: List[Tuple[Texture, int, int]] = []  # (tex, x_start, tile_w)
        self._geometry: Optional[SpectrogramGeometry] = None
        self._bouts: List[dict] = []
        self._selected_ids: List[int] = []
        self._playhead_t: float = 0.0

        # When True, an empty-area drag draws a new bout; when False, the
        # touch is passed through to the ScrollView so it can pan/fling
        # (momentum) horizontally.  Toggled from the screen's Pan/Add button.
        self.add_mode: bool = False

        # Drag / add state
        self._drag_type: Optional[str] = None   # 'onset' | 'offset' | 'add' | 'select'
        self._drag_bout_id: Optional[int] = None
        self._drag_pre_onset: float = 0.0
        self._drag_pre_offset: float = 0.0
        self._draft_onset: Optional[float] = None
        self._draft_offset: Optional[float] = None
        # Select-tap tracking
        self._drag_start_lx: float = 0.0
        self._drag_extend: bool = False   # was Shift held at touch_down?
        self._drag_cancelled: bool = False

        # Callbacks wired by ChatterScreen
        self.on_onset_live: Optional[Callable[[float], None]] = None
        self.on_offset_live: Optional[Callable[[float], None]] = None
        self.on_bout_updated: Optional[Callable[[int, float, float], None]] = None
        self.on_bout_added: Optional[Callable[[float, float], None]] = None
        self.on_seek: Optional[Callable[[float], None]] = None
        self.on_bout_selected: Optional[Callable[[int, bool], None]] = None

        self.bind(size=self._on_size_change)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_base_tiles(self, tiles: list, geometry: SpectrogramGeometry):
        """Upload RGBA tile arrays → GPU textures, set widget size, redraw.

        Must be called on the Kivy main thread.
        ``tiles`` is the list returned by ``render_spectrogram_tiles``:
        ``[(rgba_array, x_start_px, tile_w_px), ...]``.
        """
        self._textures = []
        for rgba, x_start, tile_w in tiles:
            h, w = rgba.shape[:2]
            tex = Texture.create(size=(w, h), colorfmt='rgba')
            # Kivy expects bottom-left origin; numpy array is top-left.
            # Flip vertically before uploading so the image appears right-side-up.
            rgba_flipped = rgba[::-1, :, :].tobytes()
            tex.blit_buffer(rgba_flipped, colorfmt='rgba', bufferfmt='ubyte')
            self._textures.append((tex, x_start, tile_w))

        self._geometry = geometry
        # Only override width; height fills the ScrollView via size_hint_y=1.
        self.width = geometry.total_w_px
        self._full_redraw()

    def update_bouts(self, bouts: list, selected_ids: list):
        self._bouts = bouts
        self._selected_ids = list(selected_ids)
        self._redraw_overlay()

    def set_playhead(self, t: float):
        self._playhead_t = t
        self._redraw_overlay()

    def clear(self):
        self._textures = []
        self._geometry = None
        self._bouts = []
        self._selected_ids = []
        self._playhead_t = 0.0
        self.canvas.clear()
        self.canvas.after.clear()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _full_redraw(self):
        """Redraw base tiles then overlay."""
        self.canvas.clear()
        h = self.height or _SPEC_HEIGHT_PX
        with self.canvas:
            for tex, x_start, tile_w in self._textures:
                Rectangle(texture=tex, pos=(x_start, 0), size=(tile_w, h))
        self._redraw_overlay()

    def _redraw_overlay(self):
        """Redraw only the overlay layer (bout spans, lines, playhead)."""
        self.canvas.after.clear()
        g = self._geometry
        if g is None:
            return
        h = self.height or _SPEC_HEIGHT_PX

        with self.canvas.after:
            selected_set = set(self._selected_ids)

            for i, bout in enumerate(self._bouts):
                is_outlier = bool(bout.get('outlier_flag', 0))
                is_selected = i in selected_set

                x_on = g.time_to_x(bout['onset'])
                x_off = g.time_to_x(bout['offset'])

                # Span fill
                if is_selected:
                    Color(*self._COL_SELECTED_SPAN)
                elif is_outlier:
                    Color(*self._COL_OUTLIER_SPAN)
                else:
                    Color(*self._COL_NORMAL_SPAN)
                Rectangle(pos=(x_on, 0), size=(max(x_off - x_on, 1), h))

                # Onset vertical line
                Color(*(self._COL_ONSET_OUTLIER if is_outlier else self._COL_ONSET_NORMAL))
                Line(points=[x_on, 0, x_on, h], width=2)

                # Offset vertical line (always blue)
                Color(*self._COL_OFFSET)
                Line(points=[x_off, 0, x_off, h], width=2)

                # Bout number label — upper-right (offset side), clear of the
                # top time-axis strip, so it no longer covers low-freq song.
                tag = f'Bout {i}'
                core = CoreLabel(text=tag, font_size=_BOUT_FONT, bold=True)
                core.refresh()
                tex = core.texture
                tw, th = tex.size
                ly = h - _AXIS_H - th - 4
                # Right-align at the offset line; clamp so narrow bouts don't
                # spill past their own onset.
                lx = max(x_off - tw - 6, x_on + 2)
                Color(0.0, 0.0, 0.0, 0.65)
                Rectangle(pos=(lx - 4, ly - 2), size=(tw + 8, th + 4))
                Color(1.0, 0.85, 0.85, 1.0) if is_outlier else Color(1.0, 1.0, 1.0, 1.0)
                Rectangle(texture=tex, pos=(lx, ly), size=(tw, th))

            # Draft region (during click-drag-to-add)
            if self._draft_onset is not None and self._draft_offset is not None:
                d0 = g.time_to_x(min(self._draft_onset, self._draft_offset))
                d1 = g.time_to_x(max(self._draft_onset, self._draft_offset))
                Color(*self._COL_DRAFT[:3], 0.3)
                Rectangle(pos=(d0, 0), size=(max(d1 - d0, 1), h))
                Color(*self._COL_DRAFT)
                Line(points=[d0, 0, d0, h], width=2)
                Line(points=[d1, 0, d1, h], width=2)

            # Playhead
            if self._playhead_t > 0:
                Color(*self._COL_PLAYHEAD)
                xp = g.time_to_x(self._playhead_t)
                Line(points=[xp, 0, xp, h], width=2)

            # Time-axis ruler (drawn last so labels sit on top)
            self._draw_time_axis(g, h)

    def _draw_time_axis(self, g: SpectrogramGeometry, h: float):
        """Draw a top ruler: minor ticks, major ticks + second labels,
        and faint full-height gridlines at the major ticks."""
        if g.duration <= 0 or g.total_w_px <= 0:
            return

        px_per_sec = g.total_w_px / g.duration
        # Choose a major step that yields a label roughly every ~90 px.
        major = _nice_step(90.0 / px_per_sec) if px_per_sec > 0 else 1.0
        minor = g.minor_tick_step if g.minor_tick_step and g.minor_tick_step > 0 else major / 5.0

        axis_y = h - _AXIS_H  # bottom edge of the top axis strip

        # Dark strip behind the labels for readability
        Color(0.0, 0.0, 0.0, 0.7)
        Rectangle(pos=(0, axis_y), size=(g.total_w_px, _AXIS_H))

        # Minor ticks (skip if they'd be denser than ~4 px apart)
        if minor * px_per_sec >= 4:
            Color(0.6, 0.6, 0.6, 0.8)
            n = int(g.duration / minor) + 1
            for i in range(n + 1):
                x = g.time_to_x(i * minor)
                # Tick hangs downward from the strip bottom edge
                Line(points=[x, axis_y, x, axis_y - 6], width=1)

        # Major ticks + gridlines + labels
        n = int(g.duration / major) + 1
        for i in range(n + 1):
            t = i * major
            x = g.time_to_x(t)
            # Faint gridline down through the whole spectrogram
            Color(1, 1, 1, 0.12)
            Line(points=[x, 0, x, axis_y], width=1)
            # Major tick hangs further down from strip bottom
            Color(0.95, 0.95, 0.95, 0.95)
            Line(points=[x, axis_y - 14, x, h], width=1.5)
            # Label sits inside the dark strip
            label = f'{t:g}s'
            core = CoreLabel(text=label, font_size=_TICK_FONT, bold=True)
            core.refresh()
            tex = core.texture
            tw, th = tex.size
            Rectangle(texture=tex, pos=(x + 4, axis_y + (_AXIS_H - th) // 2), size=(tw, th))

    def _on_size_change(self, *_):
        if self._textures:
            self._full_redraw()

    # ------------------------------------------------------------------
    # Touch handling
    # ------------------------------------------------------------------

    def on_touch_down(self, touch):
        # to_local converts window coordinates → widget-local coordinates,
        # which is essential when this widget is inside a ScrollView.
        lx, ly = self.to_local(*touch.pos)
        if not (0 <= lx <= self.width and 0 <= ly <= self.height):
            return False
        if self._geometry is None:
            return False

        g = self._geometry

        # Cmd (Mac) or Ctrl (Win/Linux) + drag near a boundary line → edit it.
        # Without the modifier a plain drag falls through to the ScrollView.
        cmd_ctrl_held = 'meta' in Window.modifiers or 'ctrl' in Window.modifiers
        if cmd_ctrl_held and len(self._selected_ids) == 1:
            bid = self._selected_ids[0]
            if bid < len(self._bouts):
                bout = self._bouts[bid]
                x_on = g.time_to_x(bout['onset'])
                x_off = g.time_to_x(bout['offset'])

                if abs(lx - x_on) <= _GRAB_PX:
                    touch.grab(self)
                    self._drag_type = 'onset'
                    self._drag_bout_id = bid
                    self._drag_pre_onset = bout['onset']
                    self._drag_pre_offset = bout['offset']
                    return True

                if abs(lx - x_off) <= _GRAB_PX:
                    touch.grab(self)
                    self._drag_type = 'offset'
                    self._drag_bout_id = bid
                    self._drag_pre_onset = bout['onset']
                    self._drag_pre_offset = bout['offset']
                    return True

        # Check if the tap landed inside a bout span → select it.
        # This runs before the add/shift check so a click on a span is always
        # treated as selection, not add.  Cmd/Ctrl boundary-drag takes priority
        # (handled above).  A drag > _SELECT_DRAG_PX cancels the selection so
        # the user can still scroll by starting from inside a span.
        shift_held = 'shift' in Window.modifiers
        if not cmd_ctrl_held:
            for i, bout in enumerate(self._bouts):
                x_on = g.time_to_x(bout['onset'])
                x_off = g.time_to_x(bout['offset'])
                if x_on <= lx <= x_off:
                    touch.grab(self)
                    self._drag_type = 'select'
                    self._drag_bout_id = i
                    self._drag_start_lx = lx
                    self._drag_extend = shift_held
                    self._drag_cancelled = False
                    return True

        # Empty area.  A plain drag falls through to the ScrollView so it can
        # scroll/fling with momentum.  Holding Shift turns the drag into a
        # click-drag-to-add gesture.
        if not (self.add_mode or shift_held):
            return False

        touch.grab(self)
        t = g.x_to_time(lx)
        self._drag_type = 'add'
        self._draft_onset = t
        self._draft_offset = t
        self._redraw_overlay()
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return False
        g = self._geometry
        lx, ly = self.to_local(*touch.pos)
        t = g.x_to_time(lx)

        if self._drag_type == 'onset':
            self._bouts[self._drag_bout_id]['onset'] = t
            if self.on_onset_live:
                self.on_onset_live(t)
            self._redraw_overlay()

        elif self._drag_type == 'offset':
            self._bouts[self._drag_bout_id]['offset'] = t
            if self.on_offset_live:
                self.on_offset_live(t)
            self._redraw_overlay()

        elif self._drag_type == 'add':
            self._draft_offset = t
            onset = min(self._draft_onset, self._draft_offset)
            offset = max(self._draft_onset, self._draft_offset)
            if self.on_onset_live:
                self.on_onset_live(onset)
            if self.on_offset_live:
                self.on_offset_live(offset)
            self._redraw_overlay()

        elif self._drag_type == 'select':
            if abs(lx - self._drag_start_lx) > _SELECT_DRAG_PX:
                self._drag_cancelled = True

        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        touch.ungrab(self)

        g = self._geometry
        lx, ly = self.to_local(*touch.pos)
        t = g.x_to_time(lx)
        drag_type = self._drag_type
        self._drag_type = None

        if drag_type == 'onset':
            # Restore temp mutation; let controller validate and commit
            self._bouts[self._drag_bout_id]['onset'] = self._drag_pre_onset
            if self.on_bout_updated:
                self.on_bout_updated(self._drag_bout_id, t, self._drag_pre_offset)
            self._redraw_overlay()

        elif drag_type == 'offset':
            self._bouts[self._drag_bout_id]['offset'] = self._drag_pre_offset
            if self.on_bout_updated:
                self.on_bout_updated(self._drag_bout_id, self._drag_pre_onset, t)
            self._redraw_overlay()

        elif drag_type == 'add':
            onset = min(self._draft_onset, self._draft_offset)
            offset = max(self._draft_onset, self._draft_offset)
            self._draft_onset = None
            self._draft_offset = None
            self._redraw_overlay()

            # Distinguish a meaningful drag from a tiny click
            if offset - onset > 0.05:
                if self.on_bout_added:
                    self.on_bout_added(onset, offset)
            else:
                # Treat as a seek click
                if self.on_seek:
                    self.on_seek(onset)

        elif drag_type == 'select':
            if not self._drag_cancelled and self.on_bout_selected:
                self.on_bout_selected(self._drag_bout_id, self._drag_extend)

        return True
