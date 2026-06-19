"""BandFilterSlider — a vertical, two-handle range slider for the band-pass
filter cutoffs, sized to sit to the left of the spectrogram.

The track maps **linearly** from 0 Hz (bottom) to the Nyquist frequency
``sr / 2`` (top), matching the spectrogram's linear STFT frequency axis. The
lower handle is the high-pass cutoff (energy below it is removed); the upper
handle is the low-pass cutoff (energy above it is removed). The shaded band
between them is the pass-band.

Because re-running the filter is expensive (it reloads + re-filters audio and
re-detects bouts), the widget does not fire on every drag step — it invokes the
``release_callback`` once, when a drag ends. Callers can read ``highpass_value``
and ``lowpass_value`` (Hz) at that point.
"""

from __future__ import annotations

from typing import Callable, Optional

from kivy.clock import Clock
from kivy.graphics import Color, Line, Ellipse
from kivy.uix.widget import Widget

_HANDLE_R = 10         # handle radius (px)
_GRAB_PX = 28          # touch hit-test radius around a handle
_TRACK_W = 6           # track line thickness (px)


class BandFilterSlider(Widget):
    """Vertical two-handle band-pass slider (high-pass + low-pass cutoffs)."""

    def __init__(self, sr: int = 22050, highpass: Optional[float] = 500.0,
                 lowpass: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self._sr = max(int(sr), 1)
        self._nyquist = self._sr / 2.0
        # None lowpass means "full bandwidth" → park the handle at Nyquist.
        self._hp = self._clamp(highpass if highpass is not None else 0.0)
        self._lp = self._clamp(lowpass if lowpass is not None else self._nyquist)
        # callback fired once when a drag ends
        self.release_callback: Optional[Callable[[], None]] = None
        self._drag: Optional[str] = None   # 'hp' | 'lp' | None
        self.bind(pos=lambda *_: self._redraw(), size=lambda *_: self._redraw())
        self._redraw()
        # The first _redraw above runs with the widget's default (100, 100)
        # geometry, before the parent layout assigns the real size — without
        # this the slider only appears after a manual window resize. Redraw once
        # more on the next frame, when the layout has settled.
        Clock.schedule_once(self._redraw, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def highpass_value(self) -> Optional[float]:
        """High-pass cutoff in Hz, or None when parked at 0 (disabled)."""
        v = round(self._hp, 1)
        return None if v <= 0 else v

    @property
    def lowpass_value(self) -> Optional[float]:
        """Low-pass cutoff in Hz, or None when parked at Nyquist (disabled)."""
        v = round(self._lp, 1)
        return None if v >= self._nyquist else v

    def set_sr(self, sr: int):
        """Rescale the frequency axis for a new recording's sample rate.

        Handles parked at the disabled extremes (0 / Nyquist) stay disabled;
        an active low-pass is clamped into the new range.
        """
        new_ny = max(int(sr), 1) / 2.0
        was_lp_full = self._lp >= self._nyquist
        self._sr = max(int(sr), 1)
        self._nyquist = new_ny
        if was_lp_full:
            self._lp = self._nyquist
        self._hp = self._clamp(self._hp)
        self._lp = self._clamp(self._lp)
        self._redraw()

    def set_values(self, highpass: Optional[float], lowpass: Optional[float]):
        """Set both cutoffs (Hz); None disables the respective filter."""
        self._hp = self._clamp(highpass if highpass is not None else 0.0)
        self._lp = self._clamp(lowpass if lowpass is not None else self._nyquist)
        if self._hp > self._lp:
            self._hp = self._lp
        self._redraw()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _clamp(self, hz: float) -> float:
        return max(0.0, min(self._nyquist, float(hz)))

    def _track_bounds(self):
        """(y_bottom, y_top) of the usable track in widget coords."""
        pad = _HANDLE_R + 2
        return self.y + pad, self.top - pad

    def _value_to_y(self, hz: float) -> float:
        y0, y1 = self._track_bounds()
        if self._nyquist <= 0:
            return y0
        return y0 + (hz / self._nyquist) * (y1 - y0)

    def _y_to_value(self, py: float) -> float:
        y0, y1 = self._track_bounds()
        if y1 <= y0:
            return 0.0
        frac = (py - y0) / (y1 - y0)
        return self._clamp(frac * self._nyquist)

    # ------------------------------------------------------------------
    # Touch handling
    # ------------------------------------------------------------------

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        hp_y = self._value_to_y(self._hp)
        lp_y = self._value_to_y(self._lp)
        d_hp = abs(touch.y - hp_y)
        d_lp = abs(touch.y - lp_y)
        if min(d_hp, d_lp) > _GRAB_PX:
            return False
        # Grab the nearer handle; on a tie prefer the high-pass handle.
        self._drag = 'hp' if d_hp <= d_lp else 'lp'
        touch.grab(self)
        self._apply_drag(touch.y)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self or self._drag is None:
            return False
        self._apply_drag(touch.y)
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        touch.ungrab(self)
        self._drag = None
        if self.release_callback:
            self.release_callback()
        return True

    def _apply_drag(self, py: float):
        val = self._y_to_value(py)
        if self._drag == 'hp':
            # High-pass cannot exceed the low-pass handle.
            self._hp = min(val, self._lp)
        elif self._drag == 'lp':
            # Low-pass cannot drop below the high-pass handle.
            self._lp = max(val, self._hp)
        self._redraw()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self, *_):
        self.canvas.clear()
        y0, y1 = self._track_bounds()
        cx = self.center_x
        hp_y = self._value_to_y(self._hp)
        lp_y = self._value_to_y(self._lp)
        with self.canvas:
            # Inactive track (full range)
            Color(0.30, 0.30, 0.30, 1)
            Line(points=[cx, y0, cx, y1], width=_TRACK_W / 2.0, cap='round')
            # Active pass-band between the handles
            Color(0.25, 0.55, 0.85, 1)
            Line(points=[cx, hp_y, cx, lp_y], width=_TRACK_W / 2.0, cap='round')
            # Handles: high-pass (lower) and low-pass (upper)
            Color(0.90, 0.90, 0.95, 1)
            Ellipse(pos=(cx - _HANDLE_R, hp_y - _HANDLE_R),
                    size=(_HANDLE_R * 2, _HANDLE_R * 2))
            Ellipse(pos=(cx - _HANDLE_R, lp_y - _HANDLE_R),
                    size=(_HANDLE_R * 2, _HANDLE_R * 2))
