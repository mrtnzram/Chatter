"""ParamInput — a validated float TextInput that commits on Enter / focus-loss.

Usage::

    pi = ParamInput(default=0.5, label='MFCC Thresh')
    pi.on_commit = lambda val: do_something(val)
"""

from __future__ import annotations

from typing import Callable, Optional

from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

Builder.load_string("""
<ParamInput>:
    orientation: 'horizontal'
    spacing: 10
    Label:
        id: param_label
        text: root.label_text
        size_hint_x: 1
        font_size: sp(15)
        bold: True
        color: 0.85, 0.85, 0.85, 1
        halign: 'right'
        valign: 'middle'
        text_size: self.size
    TextInput:
        id: text_input
        text: root._display_text
        multiline: False
        input_filter: 'float'
        size_hint_x: None
        width: 100
        font_size: sp(16)
        padding: [8, max((self.height - self.line_height) / 2.0, 0), 8, 0]
        on_text_validate: root._on_commit()
        on_focus: if not args[1]: root._on_commit()
        background_color: 0.12, 0.12, 0.12, 1
        foreground_color: 1, 1, 1, 1
        cursor_color: 1, 1, 1, 1
""")


class ParamInput(BoxLayout):
    """Labelled float TextInput with commit-on-enter/focus-loss."""

    def __init__(self, default: float = 0.0, label: str = '', **kwargs):
        self.label_text = label
        self._value = default
        self._display_text = str(default)
        super().__init__(**kwargs)
        self.on_commit: Optional[Callable[[float], None]] = None

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float):
        self._value = v
        self._display_text = f'{v:.4g}'
        if hasattr(self, 'ids') and 'text_input' in self.ids:
            self.ids.text_input.text = self._display_text

    def _on_commit(self):
        raw = self.ids.text_input.text.strip()
        try:
            new_val = float(raw)
        except ValueError:
            # Revert to last good value
            self.ids.text_input.text = f'{self._value:.4g}'
            return
        self._value = new_val
        if self.on_commit:
            self.on_commit(new_val)
