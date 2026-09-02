"""WelcomeScreen — landing page shown on first launch.

The user picks three directories before entering the main app:
  1. Recording directory  (where .wav files live)
  2. CSV export directory (where <recording-dir-name>.csv + .duckdb are written)
  3. Bouts audio directory (where exported audio clips are saved)

Changing the recording directory auto-fills the other two as sensible
defaults (parent dir and parent/bouts_audio); the user can override them.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from kivy.graphics import Color as GColor, Rectangle as GRect
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView, FileSystemLocal
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.metrics import dp, sp

_screen_dir = os.path.dirname(os.path.abspath(__file__))
# When frozen by PyInstaller, assets are extracted to sys._MEIPASS/assets;
# from source they live at chatter_app/assets (the parent of screens/).
_assets_dir = os.path.join(
    getattr(sys, '_MEIPASS', os.path.dirname(_screen_dir)), 'assets'
)


class _SafeFileSystem(FileSystemLocal):
    """FileChooser filesystem that never calls into pywin32.

    Kivy's FileSystemLocal.is_hidden() shells out to win32file's
    GetFileAttributesExW on Windows, and the FileChooser calls it once per
    entry, unguarded, while listing a directory. That raises on cloud-synced
    placeholders (Box/OneDrive), network shares and long paths — taking the
    whole Browse popup down with it — and needs pywin32 present at all, which
    is not a dependency we ship.

    A dotfile check behaves identically on macOS and is close enough on
    Windows: at worst a hidden folder stays visible in the picker.
    """

    def is_hidden(self, fn):
        return os.path.basename(fn).startswith('.')


class WelcomeScreen(Screen):
    """Landing page: logo placeholder + directory pickers + Launch button."""

    def __init__(self, on_launch: Callable[[str, str, str, dict], None], **kwargs):
        super().__init__(**kwargs)
        self._on_launch_cb = on_launch
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Dark background
        with self.canvas.before:
            GColor(0.08, 0.08, 0.10, 1)
            self._bg = GRect(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_bg, size=self._sync_bg)

        outer = AnchorLayout(anchor_x='center', anchor_y='center')
        self.add_widget(outer)

        # Fixed-width content column; height auto-expands to children
        content = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            width=dp(640),
            spacing=dp(16),
            padding=(0, dp(20)),
        )
        content.bind(minimum_height=content.setter('height'))
        outer.add_widget(content)

        # ── Logo ───────────────────────────────────────────────────────
        logo_anchor = AnchorLayout(size_hint_y=None, height=dp(130), anchor_x='center')
        logo_img = Image(
            source=os.path.join(_assets_dir, 'zebrafinch.png'),
            size_hint=(None, None),
            size=(dp(120), dp(120)),
            fit_mode='contain',
        )
        logo_anchor.add_widget(logo_img)
        content.add_widget(logo_anchor)

        # ── Title + subtitle ───────────────────────────────────────────
        title = Label(
            text='Chatter',
            font_size=sp(40),
            bold=True,
            size_hint_y=None, height=dp(54),
            color=(0.92, 0.93, 0.96, 1),
            halign='center',
        )
        title.bind(size=title.setter('text_size'))
        content.add_widget(title)

        subtitle = Label(
            text='Bird Song Bout Segmentation',
            font_size=sp(16),
            size_hint_y=None, height=dp(24),
            color=(0.50, 0.55, 0.65, 1),
            halign='center',
        )
        subtitle.bind(size=subtitle.setter('text_size'))
        content.add_widget(subtitle)

        content.add_widget(_thin_divider())

        # ── Directory pickers ──────────────────────────────────────────
        section_lbl = Label(
            text='Choose Directories',
            font_size=sp(13),
            bold=True,
            size_hint_y=None, height=dp(26),
            color=(0.45, 0.50, 0.62, 1),
            halign='left',
        )
        section_lbl.bind(size=section_lbl.setter('text_size'))
        content.add_widget(section_lbl)

        default = os.path.expanduser('~')
        self._rec_input   = _dir_row(content, 'Recording Directory:',   default,                              self._browse_rec)
        self._csv_input   = _dir_row(content, 'CSV Export Directory:',  default,                              self._browse_csv)
        self._audio_input = _dir_row(content, 'Bouts Audio Directory:', os.path.join(default, 'bouts_audio'), self._browse_audio)

        # Auto-fill csv / audio dirs when the recording dir changes
        self._rec_input.bind(text=self._on_rec_dir_changed)

        content.add_widget(_thin_divider())

        # ── Audio feature settings ─────────────────────────────────────
        audio_lbl = Label(
            text='Audio Feature Settings  (defaults work for most recordings)',
            font_size=sp(13),
            bold=True,
            size_hint_y=None, height=dp(26),
            color=(0.45, 0.50, 0.62, 1),
            halign='left',
        )
        audio_lbl.bind(size=audio_lbl.setter('text_size'))
        content.add_widget(audio_lbl)

        arow1 = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(10))
        self._sr_input    = _num_field(arow1, 'Sample Rate:', 44100)
        self._nmfcc_input = _num_field(arow1, 'MFCC Count:',  13)
        content.add_widget(arow1)

        arow2 = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(10))
        self._hop_input   = _num_field(arow2, 'Hop Length:',   512)
        self._frame_input = _num_field(arow2, 'Frame Length:', 2048)
        content.add_widget(arow2)

        # Band-pass cutoffs (Hz) applied before detection/display. Low-pass is
        # optional — leave blank to disable it (full bandwidth).
        arow3 = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(10))
        self._hp_input = _num_field(arow3, 'High-pass (Hz):', 500)
        self._lp_input = _num_field(arow3, 'Low-pass (Hz):', None, hint='off')
        content.add_widget(arow3)

        content.add_widget(_thin_divider())

        # ── Launch button ──────────────────────────────────────────────
        launch_anchor = AnchorLayout(size_hint_y=None, height=dp(64), anchor_x='center')
        self._launch_btn = Button(
            text='Launch',
            size_hint=(None, None),
            size=(dp(210), dp(52)),
            font_size=sp(20),
            bold=True,
            background_color=(0.10, 0.48, 0.10, 1),
            background_normal='',
        )
        self._launch_btn.bind(on_release=self._on_launch)
        launch_anchor.add_widget(self._launch_btn)
        content.add_widget(launch_anchor)

        # ── Status / error label ───────────────────────────────────────
        self._status = Label(
            text='',
            font_size=sp(14),
            size_hint_y=None, height=dp(26),
            color=(1, 0.45, 0.45, 1),
            halign='center',
        )
        self._status.bind(size=self._status.setter('text_size'))
        content.add_widget(self._status)

    def _sync_bg(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size

    # ------------------------------------------------------------------
    # Auto-derive csv / audio dirs from recording dir
    # ------------------------------------------------------------------

    def _on_rec_dir_changed(self, _inst, value: str):
        value = value.rstrip('/\\')
        if not value:
            return
        parent = os.path.dirname(value)
        if not parent or parent == value:
            parent = value
        self._csv_input.text   = parent
        self._audio_input.text = os.path.join(parent, 'bouts_audio')

    # ------------------------------------------------------------------
    # Browse popups
    # ------------------------------------------------------------------

    def _browse_rec(self, *_):
        self._open_dir_popup('Select Recording Directory', self._rec_input)

    def _browse_csv(self, *_):
        self._open_dir_popup('Select CSV Export Directory', self._csv_input)

    def _browse_audio(self, *_):
        self._open_dir_popup('Select Bouts Audio Directory', self._audio_input)

    def _open_dir_popup(self, title: str, target: TextInput):
        start = target.text if os.path.isdir(target.text) else os.path.expanduser('~')

        layout = BoxLayout(orientation='vertical', spacing=dp(6), padding=dp(6))

        hint = Label(
            text='Single-click a folder to select it  ·  Double-click to navigate into it',
            size_hint_y=None, height=dp(26),
            font_size=sp(12),
            color=(0.55, 0.60, 0.70, 1),
        )
        layout.add_widget(hint)

        chooser = FileChooserListView(path=start, dirselect=True,
                                      file_system=_SafeFileSystem())
        layout.add_widget(chooser)

        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(10))
        cancel_btn = Button(
            text='Cancel',
            background_color=(0.22, 0.22, 0.24, 1),
            background_normal='',
        )
        select_btn = Button(
            text='Select Folder',
            background_color=(0.10, 0.42, 0.10, 1),
            background_normal='',
        )
        btn_row.add_widget(cancel_btn)
        btn_row.add_widget(select_btn)
        layout.add_widget(btn_row)

        popup = Popup(
            title=title,
            content=layout,
            size_hint=(0.88, 0.82),
        )

        def on_select(*_):
            path = chooser.selection[0] if chooser.selection else chooser.path
            if path:
                target.text = path
            popup.dismiss()

        def on_cancel(*_):
            popup.dismiss()

        select_btn.bind(on_release=on_select)
        cancel_btn.bind(on_release=on_cancel)
        popup.open()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _on_launch(self, *_):
        rec_dir   = self._rec_input.text.strip()
        csv_dir   = self._csv_input.text.strip()
        audio_dir = self._audio_input.text.strip()

        if not rec_dir:
            self._set_error('Please choose a Recording Directory.')
            return
        if not os.path.isdir(rec_dir):
            self._set_error(f'Recording directory not found:\n{rec_dir}')
            return
        if not csv_dir:
            self._set_error('Please choose a CSV Export Directory.')
            return

        try:
            audio_params = {
                'sr':           int(self._sr_input.text),
                'n_mfcc':       int(self._nmfcc_input.text),
                'hop_length':   int(self._hop_input.text),
                'frame_length': int(self._frame_input.text),
            }
        except ValueError:
            self._set_error('Audio feature settings must be whole numbers.')
            return
        if any(v <= 0 for v in audio_params.values()):
            self._set_error('Audio feature settings must be positive.')
            return

        # Band-pass cutoffs: high-pass defaults, low-pass optional (blank = off).
        try:
            hp_text = self._hp_input.text.strip()
            lp_text = self._lp_input.text.strip()
            highpass = int(hp_text) if hp_text else None
            lowpass = int(lp_text) if lp_text else None
        except ValueError:
            self._set_error('Filter cutoffs must be whole numbers, or blank.')
            return
        if (highpass is not None and highpass <= 0) or \
           (lowpass is not None and lowpass <= 0):
            self._set_error('Filter cutoffs must be positive.')
            return
        if highpass is not None and lowpass is not None and highpass >= lowpass:
            self._set_error('High-pass cutoff must be below the low-pass cutoff.')
            return
        audio_params['highpass_cutoff'] = highpass
        audio_params['lowpass_cutoff'] = lowpass

        self._status.color = (0.65, 0.90, 0.65, 1)
        self._status.text  = 'Initializing...'
        self._launch_btn.disabled = True
        self._on_launch_cb(rec_dir, csv_dir, audio_dir, audio_params)

    def reset(self):
        """Restore the screen to its ready state (called when returning from a project)."""
        self._launch_btn.disabled = False
        self._status.text  = ''
        # Directory inputs are intentionally left as-is so the user can
        # quickly re-launch the same project or switch to another nearby one.

    def _set_error(self, msg: str):
        self._status.color = (1, 0.45, 0.45, 1)
        self._status.text  = msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thin_divider() -> Widget:
    d = Widget(size_hint_y=None, height=dp(1))
    with d.canvas:
        GColor(0.22, 0.22, 0.28, 1)
        rect = GRect(pos=d.pos, size=d.size)
    d.bind(pos=lambda *_: setattr(rect, 'pos', d.pos),
           size=lambda *_: setattr(rect, 'size', d.size))
    return d


def _num_field(row: BoxLayout, label_text: str, default,
               hint: str = '') -> TextInput:
    """Append a labelled integer field to *row*. Returns the TextInput.

    *default* may be None to start the field blank (used for optional settings
    like the low-pass cutoff, where blank means "disabled").
    """
    lbl = Label(
        text=label_text,
        size_hint_x=None, width=dp(110),
        font_size=sp(13),
        color=(0.75, 0.78, 0.84, 1),
        halign='right', valign='middle',
    )
    lbl.bind(size=lbl.setter('text_size'))

    ti = TextInput(
        text='' if default is None else str(default),
        hint_text=hint,
        multiline=False,
        input_filter='int',
        font_size=sp(13),
        size_hint_x=1,
        size_hint_y=1,
        background_color=(0.11, 0.11, 0.13, 1),
        foreground_color=(0.95, 0.95, 0.95, 1),
        cursor_color=(1, 1, 1, 1),
    )

    row.add_widget(lbl)
    row.add_widget(ti)
    return ti


def _dir_row(parent: BoxLayout, label_text: str,
             default: str, browse_cb) -> TextInput:
    """Append a labelled directory-picker row to *parent*. Returns the TextInput."""
    row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))

    lbl = Label(
        text=label_text,
        size_hint_x=None, width=dp(200),
        font_size=sp(14),
        color=(0.75, 0.78, 0.84, 1),
        halign='right', valign='middle',
    )
    lbl.bind(size=lbl.setter('text_size'))

    ti = TextInput(
        text=default,
        multiline=False,
        font_size=sp(13),
        size_hint_x=1,
        size_hint_y=1,
        background_color=(0.11, 0.11, 0.13, 1),
        foreground_color=(0.95, 0.95, 0.95, 1),
        cursor_color=(1, 1, 1, 1),
    )

    browse_btn = Button(
        text='Browse',
        size_hint_x=None, width=dp(90),
        size_hint_y=1,
        font_size=sp(13),
        background_color=(0.18, 0.28, 0.42, 1),
        background_normal='',
    )
    browse_btn.bind(on_release=browse_cb)

    row.add_widget(lbl)
    row.add_widget(ti)
    row.add_widget(browse_btn)
    parent.add_widget(row)
    return ti
