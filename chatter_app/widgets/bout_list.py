"""BoutList — a RecycleView-based multi-select bout list.

Each row shows "Bout N: onset–offset | Duration Xs [OUTLIER]".
Tapping a row toggles its selection.  The widget exposes:

  ``selected_ids``   — list of currently selected row indices
  ``on_selection``   — optional callback(selected_ids) fired on every change
  ``set_bouts(bouts)``  — rebuild the list from a list of bout dicts
"""

from __future__ import annotations

from typing import Callable, List, Optional

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty

Builder.load_string("""
<BoutRow>:
    canvas.before:
        Color:
            rgba: (0.25, 0.55, 1.0, 0.45) if self.is_selected else (0.15, 0.15, 0.15, 1)
        Rectangle:
            pos: self.pos
            size: self.size
    color: 1, 1, 1, 1
    text_size: self.width, None
    halign: 'left'
    valign: 'middle'
    padding: 8, 4
    size_hint_y: None
    height: 40
    font_size: sp(15)

<BoutList>:
    viewclass: 'BoutRow'
    RecycleBoxLayout:
        orientation: 'vertical'
        default_size_hint: 1, None
        default_size: None, 40
        size_hint_y: None
        height: self.minimum_height
""")


class BoutRow(RecycleDataViewBehavior, Label):
    """Single row in the bout RecycleView."""
    is_selected = BooleanProperty(False)
    index = NumericProperty(0)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        self.is_selected = data.get('selected', False)
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            shift_held = 'shift' in Window.modifiers
            self.parent.parent.select_row(self.index, extend=shift_held)
            return True
        return super().on_touch_down(touch)


class BoutList(RecycleView):
    """RecycleView that holds bout rows with multi-select support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_ids: List[int] = []
        self.on_selection: Optional[Callable[[List[int]], None]] = None
        self.data = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_bouts(self, bouts: list):
        """Rebuild rows from a list of bout dicts.  Clears current selection."""
        self.selected_ids = []
        self.data = [
            {
                'text': _bout_label(i, b),
                'selected': False,
            }
            for i, b in enumerate(bouts)
        ]

    def set_selection(self, ids: List[int]):
        """Programmatically set the selection (e.g. after a drag commits)."""
        self.selected_ids = list(ids)
        self._sync_selection_state()
        if self.on_selection:
            self.on_selection(self.selected_ids)

    def select_row(self, index: int, extend: bool = False):
        """Select a row. Without Shift: exclusive (click again to deselect).
        With Shift: add/remove from multi-selection."""
        if extend:
            if index in self.selected_ids:
                self.selected_ids.remove(index)
            else:
                self.selected_ids.append(index)
        else:
            # Single-select: deselect if already the only selection
            if self.selected_ids == [index]:
                self.selected_ids = []
            else:
                self.selected_ids = [index]
        self._sync_selection_state()
        if self.on_selection:
            self.on_selection(self.selected_ids)

    def toggle_selection(self, index: int):
        if index in self.selected_ids:
            self.selected_ids.remove(index)
        else:
            self.selected_ids.append(index)
        self._sync_selection_state()
        if self.on_selection:
            self.on_selection(self.selected_ids)

    def clear_selection(self):
        self.selected_ids = []
        self._sync_selection_state()
        if self.on_selection:
            self.on_selection([])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_selection_state(self):
        selected_set = set(self.selected_ids)
        for i, row in enumerate(self.data):
            row['selected'] = i in selected_set
        self.refresh_from_data()


def _bout_label(i: int, bout: dict) -> str:
    onset = bout.get('onset', 0.0)
    offset = bout.get('offset', 0.0)
    dur = offset - onset
    outlier = ' [OUTLIER]' if bout.get('outlier_flag', 0) else ''
    return f'Bout {i}: {onset:.2f}–{offset:.2f}s | {dur:.2f}s{outlier}'
