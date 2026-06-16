# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda create -n env_name python==3.10.16 pip
conda activate env_name
conda install -c conda-forge ipympl jupyterlab
pip install jupyter
pip install -r requirements.txt
```

Run the tool:
```bash
jupyter lab
# Then open Chatter.ipynb
```

There is no test suite and no linter configuration.

## Architecture

Chatter is a Jupyter-notebook-based tool for semi-automatic segmentation of bird song recordings into "bouts" (discrete song segments). The entry point is `Chatter.ipynb`; the three `.py` modules are imported by the notebook.

### Module responsibilities

**[audio_utils.py](audio_utils.py)** — All audio processing logic:
- `create_initial_dataset(root_dir)` — Walks a directory of `.wav` files and returns a pandas DataFrame. Requires filenames in the format `Genus-species-birdid.wav` (e.g., `Melospiza-melodia-599851.wav`); the parser splits on `-` to extract `species` and `bird_id`.
- `AudioFeatureExtractor` — The core detection engine. Its `compute_all_features(row)` runs the full pipeline: load audio → high-pass filter → normalize → compute spectral flux / MFCCs / RMS energy → detect active regions → refine with MFCC variance and energy thresholds → merge short silences → filter short bouts → flag outlier bouts via cosine distance. Optionally runs BirdNET or a custom sklearn model for post-classification. All detection parameters are attributes set on the extractor instance (not passed per-call), so `Chatter` modifies them directly when widget values change.

**[chatter_core.py](chatter_core.py)** — The interactive widget UI:
- `Chatter(df, extractor, bouts_csv=None)` — Wraps `AudioFeatureExtractor` and a `df` DataFrame in an `ipywidgets` interface. Calls `extractor.compute_all_features()` on every parameter change (re-runs the full pipeline). Spectrogram images are cached per recording index in `self.spectrogram_cache` and invalidated on bird change to avoid redundant STFT calls. Per-bird parameter state is stored in `self.bird_params` so switching recordings restores their saved values. `Export Bouts` writes audio clips to `bouts_audio/` and appends to `bouts.csv` (or whichever path is passed as `bouts_csv`). On construction, if a `bouts.csv` already exists it is loaded and reconciled back into `df['bouts']` so progress persists across sessions.

**[visualizations.py](visualizations.py)** — Plotting helpers:
- `plot_spectrogram_base_from_row` — Renders the base spectrogram (STFT) as a matplotlib figure; returns `(fig, ax, S_db, duration)` so overlays can be added to the same axes without re-computing the STFT.
- `plot_bout_overlays` — Draws bout spans (green = normal, red = outlier), onset/offset lines, and inter-bout interval brackets onto an existing axis.
- `show_scrollable_figure` — Encodes a figure as base64 PNG and renders it inside a horizontally-scrollable HTML `<div>`, which is necessary because long recordings produce very wide spectrograms.

### Data flow

```
root_dir/
  Genus-species-birdid.wav   →  create_initial_dataset()  →  df (one row per file)
                                                              ↓
                                           AudioFeatureExtractor.compute_all_features()
                                                              ↓
                                    df['bouts'] = [{onset, offset, wavstart, wavend, outlier_flag}, ...]
                                                              ↓
                                              Chatter widget (interactive editing)
                                                              ↓
                            bouts_audio/<species>_<bird_id>_bout<N>.wav  +  bouts.csv
```

### Key data structures

- **`df`** (main DataFrame): one row per recording. Key columns: `species`, `bird_id`, `wav_location`, `song_id`, `audio` (numpy array), `sr`, `bouts` (list of dicts), plus feature arrays (`mfcc`, `spectral_flux`, `rms_energy`, `active_regions`, `refined_regions`).
- **`bouts`** (list of dicts per row): each dict has `onset`, `offset` (unpadded, seconds), `wavstart`, `wavend` (padded for export), and optionally `outlier_flag` (1 = outlier), `birdnet_flag`.
- **`bouts_df`** / `bouts.csv`: flattened export; one row per bout with `species`, `bird_id`, `song_id`, `bout_id`, `duration`, `onset`, `offset`, `wavstart`, `wavend`, `intersong_interval`, `bout_wav`.

### Optional ML classifiers

- **Custom model**: pass a joblib-serialized sklearn model path to `AudioFeatureExtractor(model='path/to/model.pkl')`. Must accept feature vectors of shape `(n_mfcc * 2,)` (MFCC mean + std per bout).
- **BirdNET**: set `use_birdnet=True` and `birdnet_model_path='BirdNETmodel/...'`. Requires the `birdnetlib` package. The TFLite model files are stored in `BirdNETmodel/`.
