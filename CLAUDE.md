# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). There is no `requirements.txt`.

```bash
uv sync                  # installs all base dependencies into .venv
uv sync --extra birdnet  # also installs birdnetlib + tensorflow for BirdNET support
uv run jupyter lab       # then open Chatter.ipynb
```

`pyproject.toml` declares direct dependencies; `uv.lock` pins the full resolved set. There is no test suite and no linter configuration.

## Architecture

Chatter is a Jupyter-notebook-based tool for semi-automatic segmentation of bird song recordings into "bouts" (discrete song segments). The entry point is `Chatter.ipynb`; the four `.py` modules are imported by the notebook.

### Module responsibilities

**[audio_utils.py](audio_utils.py)** — All audio processing logic:
- `create_initial_dataset(root_dir)` — Walks a directory of `.wav` files and returns a pandas DataFrame. Requires filenames in the format `Genus-species-birdid.wav` (e.g., `Melospiza-melodia-599851.wav`); the parser splits on `-` to extract `species` and `bird_id`.
- `AudioFeatureExtractor` — The core detection engine. Its `compute_all_features(row)` runs the full pipeline: load audio → high-pass filter → normalize → compute spectral flux / MFCCs / RMS energy → detect active regions → refine with MFCC variance and energy thresholds → merge short silences → filter short bouts → flag outlier bouts via cosine distance. Optionally runs BirdNET or a custom sklearn model for post-classification. All detection parameters are attributes set on the extractor instance (not passed per-call), so `Chatter` modifies them directly when widget values change.

**[chatter_store.py](chatter_store.py)** — Data-access layer (two DuckDB connections):
- `ChatterStore(duckdb_path, csv_path, spectro_cache_cap)` — owns all storage. Never import or query DuckDB directly from `chatter_core.py`; go through this wrapper.
- **Persistent connection** (`chatter.duckdb`): `bouts` table, PK `(species, bird_id, song_id, bout_id)`. `upsert_bouts(bout_rows)` replaces all bouts for a recording on each export (delete-then-insert per `song_id`). `get_bouts_df()` queries it. `export_bouts_csv()` writes it to `bouts.csv`.
- **Ephemeral connection** (`:memory:`): `spectrogram_cache` table, gzip-compressed `np.save` BLOBs, keyed by `sha256(wav_location|sr|hop_length|frame_length)`. LRU cap (default 8 entries) is tracked in a Python `OrderedDict`; the `:memory:` connection is never written to disk and is freed on process exit or `close()`. `get_cached_spectrogram(key)` returns `(S_db, sr)` or `None` on miss; `set_cached_spectrogram(key, S_db, sr)` stores and enforces the cap.
- **CSV migration**: on first launch, if `bouts.csv` exists and the `bouts` table is empty, the CSV is imported once so prior sessions are not lost.
- `get_bouts_csv_df()` — reads `bouts.csv` from disk directly (used by `chatter.bouts_df`).

**[chatter_core.py](chatter_core.py)** — The interactive widget UI:
- `Chatter(df, extractor, bouts_csv=None, duckdb_path="chatter.duckdb")` — Wraps `AudioFeatureExtractor` and `df` in an `ipywidgets` interface. Constructs a `ChatterStore` on init; reconciles saved bouts (from DuckDB) back into `df['bouts']` grouped by `(species, bird_id, song_id)`. Per-bird parameter state is stored in `self.bird_params`.
- `chatter.bouts_df` — **property** that reads `bouts.csv` from disk (via `store.get_bouts_csv_df()`), not DuckDB. This is what the notebook previews.
- `get_cached_spectrogram(wav_location, audio, sr)` — checks the ephemeral cache first; computes and stores on miss. Cache key includes `hop_length` and `frame_length` so changing `sr` invalidates it but tweaking MFCC thresholds does not.
- `_on_save_bouts_clicked` — builds `bout_rows`, calls `store.upsert_bouts()` then `store.export_bouts_csv()`. DuckDB is the write-path; `bouts.csv` is produced as a side effect for back-compat.
- Bout-editing handlers (`_on_update_bout_clicked`, `on_add_bout_clicked`, `_on_remove_bouts_clicked`, `_on_not_outlier_clicked`) stay **pure in-memory pandas** — they mutate `self.current_bouts` / `self.df` only. Data reaches DuckDB/CSV only at Export time.
- `close()` — shuts down both DuckDB connections cleanly. Call this at the end of a notebook session.

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
                            store.upsert_bouts() → chatter.duckdb  →  store.export_bouts_csv()
                                                                               ↓
                            bouts_audio/<species>_<bird_id>_bout<N>.wav  +  bouts.csv
```

### Key data structures

- **`df`** (main DataFrame): one row per recording. Key columns: `species`, `bird_id`, `wav_location`, `song_id`, `audio` (numpy array), `sr`, `bouts` (list of dicts), plus feature arrays (`mfcc`, `spectral_flux`, `rms_energy`, `active_regions`, `refined_regions`).
- **`bouts`** (list of dicts per row): each dict has `onset`, `offset` (unpadded, seconds), `wavstart`, `wavend` (padded for export), and optionally `outlier_flag` (1 = outlier), `birdnet_flag`.
- **`bouts.csv`** / `chatter.bouts_df`: flattened export; one row per bout. Columns: `species`, `bird_id`, `wav_location`, `song_id`, `bout_id`, `duration`, `onset`, `offset`, `wavstart`, `wavend`, `intersong_interval`, `bout_wav`. The PK in DuckDB is `(species, bird_id, song_id, bout_id)` — multiple recordings of the same bird coexist via `song_id`.
- **`chatter.duckdb`**: persistent DuckDB file created in CWD at notebook runtime. Gitignored; user data, not source.

### Optional ML classifiers

- **Custom model**: pass a joblib-serialized sklearn model path to `AudioFeatureExtractor(model='path/to/model.pkl')`. Must accept feature vectors of shape `(n_mfcc * 2,)` (MFCC mean + std per bout).
- **BirdNET**: set `use_birdnet=True` and `birdnet_model_path='BirdNETmodel/...'`. Requires `uv sync --extra birdnet`. The TFLite model files are stored in `BirdNETmodel/`.
