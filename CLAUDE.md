# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

```
Chatter/
├── notebook/               # Jupyter notebook implementation (research/legacy)
│   ├── Chatter.ipynb       # Entry point
│   ├── audio_utils.py      # Audio processing + feature extraction
│   ├── chatter_core.py     # ipywidgets UI
│   ├── chatter_store.py    # DuckDB data layer
│   └── visualizations.py  # matplotlib / HTML spectrogram helpers
├── chatter_app/            # Kivy desktop application (primary)
│   ├── main.py             # App entry point (WelcomeScreen → ChatterScreen)
│   ├── core/
│   │   ├── audio_utils.py        # Same as notebook/ version
│   │   ├── chatter_controller.py # Bridges UI to store + extractor
│   │   ├── chatter_store.py      # Same as notebook/ version
│   │   └── visualizations.py    # Same as notebook/ version
│   ├── screens/
│   │   ├── welcome_screen.py     # Landing page (directory pickers)
│   │   └── chatter_screen.py    # Main analysis screen
│   └── widgets/
│       ├── bout_list.py          # RecycleView multi-select bout list
│       ├── param_input.py        # Labelled float TextInput
│       └── spectrogram_view.py  # Tiled spectrogram canvas + drag editing
├── BirdNETmodel/           # Optional TFLite models (use_birdnet=True)
├── Songs/                  # Sample WAV files for testing
├── pyproject.toml          # Project dependencies (uv / pip)
└── chatter_app/requirements_kivy.txt  # Kivy-specific deps
```

There is no test suite and no linter configuration.

---

## Setup — Jupyter notebook

```bash
# Install dependencies (uv recommended, or pip)
uv sync
# or: pip install -e .

cd notebook
jupyter lab
# Then open Chatter.ipynb
```

## Setup — Kivy desktop app

```bash
pip install -r chatter_app/requirements_kivy.txt

cd chatter_app
python main.py
```

---

## Architecture — Kivy app (`chatter_app/`)

The desktop app is the actively developed version. It opens a **WelcomeScreen** where the user selects three directories, then transitions to **ChatterScreen** for interactive bout segmentation.

### Screens

**[chatter_app/screens/welcome_screen.py](chatter_app/screens/welcome_screen.py)**
- Directory pickers for recording dir, CSV export dir, and bouts audio dir
- Auto-derives csv/audio dirs from the recording dir's parent
- Calls back into `ChatterApp._on_launch()` which spins a background thread for dataset scanning

**[chatter_app/screens/chatter_screen.py](chatter_app/screens/chatter_screen.py)**
- Full analysis UI: bird spinner, parameter rows, bout list, onset/offset editing, spectrogram
- "New Project" button (top-right) triggers a confirmation popup and returns to WelcomeScreen
- All slow operations (recompute, finalize, export) run on background threads; results marshalled back via `Clock.schedule_once()`

### Widgets

**[chatter_app/widgets/spectrogram_view.py](chatter_app/widgets/spectrogram_view.py)**
- Tiled STFT spectrogram rendered on a Kivy canvas
- Drag-to-scroll; Shift+drag adds a new bout; Cmd/Ctrl+drag moves a bout boundary

**[chatter_app/widgets/bout_list.py](chatter_app/widgets/bout_list.py)**
- `RecycleView`-based list with multi-select (Shift+click) support

**[chatter_app/widgets/param_input.py](chatter_app/widgets/param_input.py)**
- Labelled float `TextInput` that commits on Enter or focus loss

### Core

**[chatter_app/core/chatter_controller.py](chatter_app/core/chatter_controller.py)**
- Owns the `df` DataFrame, `AudioFeatureExtractor`, and `ChatterStore`
- `recompute(idx, params)` — runs the full detection pipeline for one recording
- `finalize(idx)` — marks detection parameters as final
- `export(idx, output_dir)` — writes audio clips to `output_dir/` and persists bouts to DuckDB + CSV
- `get_cached_spectrogram(wav_path, audio, sr)` — delegates to the store's in-memory STFT cache
- `close()` — must be called when switching projects to release the DuckDB file lock

**[chatter_app/core/audio_utils.py](chatter_app/core/audio_utils.py)** — same as `notebook/audio_utils.py`

**[chatter_app/core/chatter_store.py](chatter_app/core/chatter_store.py)** — same as `notebook/chatter_store.py`

### App lifecycle / multi-project

`ChatterApp` (in `main.py`) manages `ScreenManager` transitions:

1. `WelcomeScreen` is created once and stays in the manager permanently
2. On Launch: background thread scans wav files → creates `ChatterController` → adds `ChatterScreen`
3. On "New Project": fade to WelcomeScreen → after 0.4 s (transition complete) remove `ChatterScreen` and call `ctrl.close()` to release DuckDB

---

## Architecture — Jupyter notebook (`notebook/`)

The notebook implementation is the original research tool, kept for reference.

**[notebook/audio_utils.py](notebook/audio_utils.py)** — Audio processing logic:
- `create_initial_dataset(root_dir)` — Walks a directory of `.wav` files and returns a pandas DataFrame. Requires filenames in the format `Genus-species-birdid.wav`; splits on `-` to extract `species` and `bird_id`.
- `AudioFeatureExtractor` — Core detection engine. `compute_all_features(row)` runs: load audio → high-pass filter → normalize → compute spectral flux / MFCCs / RMS energy → detect active regions → refine with MFCC variance and energy thresholds → merge short silences → filter short bouts → flag outlier bouts via cosine distance. Optionally runs BirdNET or a custom sklearn model.

**[notebook/chatter_core.py](notebook/chatter_core.py)** — ipywidgets UI:
- `Chatter(df, extractor, bouts_csv=None)` — Interactive widget wrapping `AudioFeatureExtractor`. Re-runs the full pipeline on every parameter change. Spectrogram images are cached in `self.spectrogram_cache`. Per-bird parameter state is stored in `self.bird_params`. `Export Bouts` writes audio clips to `bouts_audio/` and persists to DuckDB + CSV.

**[notebook/chatter_store.py](notebook/chatter_store.py)** — DuckDB data layer:
- Two connections: persistent (`chatter.duckdb` on disk for bouts) and ephemeral (`:memory:` for spectrogram cache with LRU eviction)
- Migrates a pre-existing `bouts.csv` into DuckDB on first run

**[notebook/visualizations.py](notebook/visualizations.py)** — Plotting helpers:
- `plot_spectrogram_base_from_row` — Renders STFT spectrogram; returns `(fig, ax, S_db, duration)`
- `plot_bout_overlays` — Draws bout spans, onset/offset lines, and inter-bout interval brackets
- `show_scrollable_figure` — Encodes a figure as base64 PNG in a horizontally-scrollable HTML `<div>`

---

## Shared data structures

These are the same in both implementations:

- **`df`** (main DataFrame): one row per recording. Key columns: `species`, `bird_id`, `wav_location`, `song_id`, `audio` (numpy array), `sr`, `bouts` (list of dicts), plus feature arrays (`mfcc`, `spectral_flux`, `rms_energy`, `active_regions`, `refined_regions`).
- **`bouts`** (list of dicts per row): each dict has `onset`, `offset` (unpadded seconds), `wavstart`, `wavend` (padded for export), and optionally `outlier_flag` (1 = outlier), `birdnet_flag`.
- **`bouts.csv` / DuckDB `bouts` table**: flattened export — one row per bout with `species`, `bird_id`, `song_id`, `bout_id`, `duration`, `onset`, `offset`, `wavstart`, `wavend`, `intersong_interval`, `bout_wav`.

## Optional ML classifiers

- **Custom model**: pass a joblib-serialized sklearn model path to `AudioFeatureExtractor(model='path/to/model.pkl')`. Must accept feature vectors of shape `(n_mfcc * 2,)` (MFCC mean + std per bout).
- **BirdNET**: set `use_birdnet=True` and `birdnet_model_path='BirdNETmodel/...'`. Requires `birdnetlib`. TFLite model files are in `BirdNETmodel/`.
