"""TickSlider — a horizontal Slider that marks its default value on the track.

A small vertical tick is drawn on the track at ``default_value`` so the user can
always see where the neutral/default setting is. Double-tapping anywhere on the
slider snaps the value back to that default, giving an easy "reset" gesture.

Used for the display-only brightness/contrast sliders, whose defaults (0.0 and
1.0 respectively) are the visually-neutral settings.
"""

from __future__ import annotations

from kivy.graphics import Color, Line
from kivy.properties import NumericProperty
from kivy.uix.slider import Slider


class TickSlider(Slider):
    """Horizontal Slider with a default-value tick mark and double-tap reset."""

    default_value = NumericProperty(0.0)
    """Value the tick marks and that a double-tap resets the slider to."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            pos=self._redraw_tick, size=self._redraw_tick,
            min=self._redraw_tick, max=self._redraw_tick,
            default_value=self._redraw_tick,
        )
        self._redraw_tick()

    def _redraw_tick(self, *_):
        # Drawn on canvas.after so it sits on top of the slider's own track/cursor.
        self.canvas.after.clear()
        if self.max == self.min:
            return
        frac = (self.default_value - self.min) / (self.max - self.min)
        x0 = self.x + self.padding
        x1 = self.right - self.padding
        tx = x0 + frac * (x1 - x0)
        cy = self.center_y
        half = 9  # tick half-height (px)
        with self.canvas.after:
            Color(0.85, 0.85, 0.55, 0.9)
            Line(points=[tx, cy - half, tx, cy + half], width=1.2)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and touch.is_double_tap:
            self.value = self.default_value
            return True
        return super().on_touch_down(touch)
