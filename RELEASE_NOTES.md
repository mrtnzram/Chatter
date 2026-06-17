# Chatter — v0.3.0 (DuckDB Data Layer)

**June 16, 2026**

---

## Overview

This release replaces the CSV-based data layer with a proper database backend (DuckDB) and migrates the project to `uv` for reproducible environment management. The result is significantly lower memory usage on large datasets and a more resilient persistence model.

---

## What's New

### DuckDB Data Layer (`chatter_store.py`)

A new dedicated data-access module replaces all raw CSV/pandas operations in `chatter_core.py`. Two connections with deliberately different lifetimes handle different concerns:

- **Persistent** (`chatter.duckdb` on disk): stores the `bouts` table with a composite primary key (`species`, `bird_id`, `song_id`, `bout_id`). Survives restarts; replaces the `bouts.csv`-as-database pattern.
- **Ephemeral** (in-memory DuckDB): stores the spectrogram cache. Created fresh on every launch, never written to disk, and freed automatically on exit. Bounded by an **LRU cap of 8 entries** to prevent unbounded memory growth on large datasets.

Spectrogram cache entries are keyed by a SHA-256 hash of `(wav_location, sr, hop_length, frame_length)` and stored as gzip-compressed numpy arrays, so cache hits are fast even for long recordings.

### Automatic CSV Migration

On first launch, if `chatter.duckdb` is empty but a legacy `bouts.csv` exists, it is automatically imported into DuckDB — no manual migration step required. Existing sessions resume exactly where they left off.

### `uv`-based Environment Management

`requirements.txt` is replaced by `pyproject.toml` + a pinned `uv.lock`. Setup is now:

```bash
uv sync                       # standard
uv sync --extra birdnet       # with BirdNET support
uv run jupyter lab
```

---

## Bug Fixes & Memory Improvements (since v0.2.x)

### Garbage Collection Fixes *(March 2026)*

Four separate memory leaks in `chatter_core.py` were resolved:

- Removed eager spectrogram cache population on init (previously loaded all birds upfront)
- Added cache eviction for the previous bird on dropdown change (entries were accumulating indefinitely)
- Explicit `close()` and dereference of `current_fig`/`ax` on bird change and redraw (matplotlib figures were being retained)
- `new_bouts_df` is now freed after `concat` in `_on_save_bouts_clicked` (the old pattern stacked the same variable repeatedly)

### Configurable `bouts_csv` Path *(April 2026)*

`Chatter.__init__` now accepts an explicit `bouts_csv=` parameter. Previously the path was hardcoded to `"bouts.csv"`, making it impossible to run multiple independent sessions in the same directory. Saved bouts are also now reconciled back into `self.df` on construction so the spectrogram overlays are correct when resuming a session.

### macOS-Compatible Dependencies *(May 2026)*

`requirements.txt` was updated to remove platform-specific pins that caused installation failures on macOS.

---

## Breaking Changes

- `requirements.txt` is no longer used for installation. Run `uv sync` instead (see Getting Started in the README).
- `chatter.bouts_df` is now backed by DuckDB rather than an in-memory DataFrame. The public API (`chatter.bouts_df`, `Export Bouts`) is unchanged, but direct DataFrame mutation of `chatter.bouts_df` will no longer persist — use `Export Bouts` to save.

---

## Files Changed

| File | Change |
|---|---|
| `chatter_store.py` | New — DuckDB data-access layer |
| `chatter_core.py` | Refactored to use `ChatterStore`; memory leak fixes; `bouts_csv` param |
| `pyproject.toml` | New — `uv` project config replacing `requirements.txt` |
| `uv.lock` | New — pinned dependency lockfile |
| `Chatter.ipynb` | Simplified notebook setup cells |
| `README.md` | Updated setup instructions for `uv` |
| `.gitignore` | Added common Python/Jupyter ignores |
