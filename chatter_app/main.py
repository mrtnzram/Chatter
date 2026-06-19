"""Chatter — Kivy desktop application entry point.

Usage
-----
    cd chatter_app
    python main.py

The app opens a welcome screen where the user selects:
  - Recording directory (contains .wav files)
  - CSV export directory (<recording-dir-name>.csv + .duckdb are written here)
  - Bouts audio directory (exported audio clips go here)

BirdNET
-------
Set ``USE_BIRDNET = True`` below and provide ``BIRDNET_MODEL_PATH`` if you
have ``birdnetlib`` installed and a BirdNET TFLite model directory.
"""

import os
import sys
import threading

# ---------------------------------------------------------------------------
# Kivy environment setup — must happen before any kivy import
# ---------------------------------------------------------------------------
os.environ.setdefault('KIVY_NO_ENV_CONFIG', '1')

from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')
Config.set('graphics', 'resizable', '1')

# ---------------------------------------------------------------------------
# Path setup so core/ and widgets/ are importable
# ---------------------------------------------------------------------------
_app_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_app_dir, 'core')
_widgets_dir = os.path.join(_app_dir, 'widgets')
_screens_dir = os.path.join(_app_dir, 'screens')
for _p in (_core_dir, _widgets_dir, _screens_dir, _app_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def resource_path(*parts):
    """Resolve a bundled resource path.

    Works both when running from source (relative to this file) and when
    frozen by PyInstaller (relative to the extraction dir ``sys._MEIPASS``).
    """
    base = getattr(sys, '_MEIPASS', _app_dir)
    return os.path.join(base, *parts)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USE_BIRDNET = False
BIRDNET_MODEL_PATH = os.path.join(_app_dir, '..', 'BirdNETmodel')

# ---------------------------------------------------------------------------
# Kivy imports (after env setup)
# ---------------------------------------------------------------------------
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, FadeTransition

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from audio_utils import AudioFeatureExtractor, create_initial_dataset
from chatter_controller import ChatterController
from chatter_screen import ChatterScreen
from welcome_screen import WelcomeScreen


class ChatterApp(App):
    title = 'Chatter — Bird Song Bout Segmentation'
    icon = resource_path('assets', 'zebrafinch.png')

    def build(self):
        self._sm = ScreenManager(transition=FadeTransition(duration=0.25))
        welcome = WelcomeScreen(on_launch=self._on_launch, name='welcome')
        self._sm.add_widget(welcome)
        return self._sm

    # ------------------------------------------------------------------
    # Welcome screen → main app transition
    # ------------------------------------------------------------------

    def _on_launch(self, songs_dir: str, csv_dir: str, bouts_audio_dir: str,
                   audio_params: dict | None = None):
        """Called by WelcomeScreen after the user clicks Launch.

        Runs dataset scanning + controller setup on a background thread so the
        UI stays responsive, then switches to ChatterScreen on the main thread.

        ``audio_params`` carries the audio-feature settings chosen on the
        welcome screen (sr, n_mfcc, hop_length, frame_length).
        """
        extractor_kwargs = dict(audio_params or {})
        if USE_BIRDNET and os.path.isdir(BIRDNET_MODEL_PATH):
            extractor_kwargs['use_birdnet'] = True
            extractor_kwargs['birdnet_model_path'] = BIRDNET_MODEL_PATH
            print('[Chatter] BirdNET classification enabled.')

        extractor = AudioFeatureExtractor(**extractor_kwargs)

        def _worker():
            try:
                print(f'[Chatter] Scanning {songs_dir} for .wav files…')
                df = create_initial_dataset(songs_dir)
                if df.empty:
                    Clock.schedule_once(lambda _: self._on_launch_error(
                        f'No .wav files found in:\n{songs_dir}'
                    ), 0)
                    return

                os.makedirs(csv_dir, exist_ok=True)
                # Name the persisted store after the recording directory so each
                # recording set keeps its own bouts file (e.g. "MyBirds.csv" /
                # "MyBirds.duckdb").  Keeping them per-recording is what lets the
                # store reload and prepopulate *this* directory's already-exported
                # bouts on reopen instead of bleeding in another project's data.
                rec_name = os.path.basename(os.path.normpath(songs_dir)) or 'bouts'
                duckdb_path = os.path.join(csv_dir, f'{rec_name}.duckdb')
                csv_path    = os.path.join(csv_dir, f'{rec_name}.csv')

                ctrl = ChatterController(
                    df, extractor,
                    duckdb_path=duckdb_path,
                    csv_path=csv_path,
                )
                Clock.schedule_once(
                    lambda _: self._on_launch_done(ctrl, bouts_audio_dir), 0
                )
            except Exception as exc:
                Clock.schedule_once(
                    lambda _: self._on_launch_error(str(exc)), 0
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_launch_done(self, ctrl: ChatterController, bouts_audio_dir: str):
        self._ctrl = ctrl
        screen = ChatterScreen(
            controller=ctrl,
            bouts_audio_dir=bouts_audio_dir,
            on_back=self._go_back_to_welcome,
            name='chatter',
        )
        self._sm.add_widget(screen)
        self._sm.current = 'chatter'

    def _on_launch_error(self, msg: str):
        welcome = self._sm.get_screen('welcome')
        welcome._set_error(msg)
        welcome._launch_btn.disabled = False

    def _go_back_to_welcome(self):
        """Tear down the current project and return to the welcome screen.

        Sequence:
          1. Transition to welcome (fade) so the user sees the screen change.
          2. After the transition, remove ChatterScreen from the manager and
             close the DuckDB connection — safe because the screen is no longer
             visible and no background threads should be writing.
        """
        welcome = self._sm.get_screen('welcome')
        welcome.reset()
        self._sm.current = 'welcome'

        def _cleanup(_dt):
            if self._sm.has_screen('chatter'):
                chatter = self._sm.get_screen('chatter')
                self._sm.remove_widget(chatter)
            if hasattr(self, '_ctrl'):
                self._ctrl.close()
                del self._ctrl

        # Wait for the FadeTransition (0.25 s) to finish before cleanup.
        Clock.schedule_once(_cleanup, 0.4)

    def on_stop(self):
        if hasattr(self, '_ctrl'):
            self._ctrl.close()


def main():
    ChatterApp().run()


if __name__ == '__main__':
    main()
