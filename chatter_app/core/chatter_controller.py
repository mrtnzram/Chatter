"""UI-agnostic controller that drives all bout and feature logic.

The Kivy screen calls these methods; they never touch any widget.
Every mutation method returns (ok: bool, message: str) so the screen
can update its status bar and decide whether to redraw.
"""

import os
import sys

import numpy as np
import pandas as pd

# Allow running from the project root or from within chatter_app/
_core_dir = os.path.dirname(__file__)
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from chatter_store import ChatterStore


_DEFAULT_PARAMS = {
    'mfcc_threshold': 0.5,
    'energy_threshold': 0.1,
    'active_region_thresh': 0.001,
    'min_silence': 0.9,
    'min_bout_len': 1.0,
    'pad': 0.5,
    # Band-pass filter cutoffs (Hz). highpass_cutoff preserves the legacy 500 Hz
    # high-pass; lowpass_cutoff=None disables the low-pass (full bandwidth).
    'highpass_cutoff': 500.0,
    'lowpass_cutoff': None,
}


class ChatterController:
    def __init__(self, df, extractor, duckdb_path='chatter.duckdb', csv_path='bouts.csv'):
        self.df = df
        self.extractor = extractor
        self.store = ChatterStore(duckdb_path=duckdb_path, csv_path=csv_path)
        # idx → list of bout dicts (in-session edits, not yet persisted)
        self.current_bouts: dict = {}
        # idx → param dict
        self.bird_params: dict = {}
        # idxs that have been exported to DuckDB + WAV files this session or previously
        self.exported_idxs: set = set()
        # Ensure columns expected downstream always exist
        for col in ('audio', 'sr', 'bouts'):
            if col not in self.df.columns:
                self.df[col] = None
        self._load_saved_bouts()

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _load_saved_bouts(self):
        """Reconcile DuckDB-persisted bouts back into self.df on startup."""
        saved = self.store.get_bouts_df()
        if saved.empty:
            return
        for (species, bird_id, song_id), group in saved.groupby(['species', 'bird_id', 'song_id']):
            mask = (
                (self.df['species'] == species)
                & (self.df['bird_id'] == bird_id)
                & (self.df['song_id'] == song_id)
            )
            if not mask.any():
                continue
            idx = int(self.df[mask].index[0])
            bouts = group.sort_values('onset').apply(
                lambda r: {
                    'onset': r['onset'],
                    'offset': r['offset'],
                    'wavstart': r['wavstart'],
                    'wavend': r['wavend'],
                    'outlier_flag': 0,
                },
                axis=1,
            ).tolist()
            self.df.at[idx, 'bouts'] = bouts
            self.current_bouts[idx] = bouts
            self.exported_idxs.add(idx)

    def get_bird_options(self):
        """Return [(label, idx, is_exported), ...] for the bird Spinner."""
        options = []
        for idx, row in self.df.iterrows():
            suffix = f"_{row['chunk_num']}" if row.get('n_chunks', 1) > 1 else ''
            label = f"{row['species']} {row['bird_id']}{suffix}"
            options.append((label, idx, idx in self.exported_idxs))
        return options

    def is_exported(self, idx: int) -> bool:
        return idx in self.exported_idxs

    # ------------------------------------------------------------------
    # Per-bird parameter state
    # ------------------------------------------------------------------

    def get_params(self, idx: int) -> dict:
        if idx not in self.bird_params:
            self.bird_params[idx] = dict(_DEFAULT_PARAMS)
        return dict(self.bird_params[idx])

    def save_params(self, idx: int, params: dict):
        self.bird_params[idx] = dict(params)

    def _apply_params(self, params: dict):
        self.extractor.mfcc_threshold = params.get('mfcc_threshold', 0.5)
        self.extractor.energy_threshold_pct = params.get('energy_threshold', 0.1)
        self.extractor.active_region_threshold_pct = params.get('active_region_thresh', 0.001)
        self.extractor.min_silence = params.get('min_silence', 0.9)
        self.extractor.min_bout_length = params.get('min_bout_len', 1.0)
        self.extractor.pad = params.get('pad', 0.5)
        self.extractor.highpass_cutoff = params.get('highpass_cutoff', 500.0)
        self.extractor.lowpass_cutoff = params.get('lowpass_cutoff', None)

    def invalidate_audio(self, idx: int):
        """Drop the cached waveform so the next recompute reloads + re-filters
        from disk. Call this when a filter cutoff changes (the spectrogram
        cache is busted separately via the cutoff-aware cache key)."""
        if 'audio' in self.df.columns:
            self.df.at[idx, 'audio'] = None

    # ------------------------------------------------------------------
    # Core recompute (run on a background thread)
    # ------------------------------------------------------------------

    def recompute(self, idx: int, params: dict, force: bool = False):
        """Load audio, then return bouts.

        When force=False (default), saved bouts from a prior session or from
        the initial detection pass are reused without re-running detection.
        When force=True, detection always reruns with the current params
        (used when the user explicitly changes a detection parameter).

        Returns (bouts, features_dict).
        Call from a background thread; results are safe to read on any thread.
        """
        self.save_params(idx, params)
        self._apply_params(params)

        if force:
            self.current_bouts.pop(idx, None)
            self.df.at[idx, 'bouts'] = None

        row = self.df.iloc[idx].copy()
        # Lazy audio load — always needed for the spectrogram render.
        # For chunked recordings use offset/duration to load only the slice.
        if not isinstance(row.get('audio'), np.ndarray):
            chunk_start = float(row.get('chunk_start') or 0.0)
            chunk_end = row.get('chunk_end')
            chunk_dur = float(chunk_end - chunk_start) if chunk_end is not None else None
            audio, sr = self.extractor.load_audio(
                row['wav_location'],
                offset=chunk_start,
                duration=chunk_dur,
            )
            self.df.at[idx, 'audio'] = audio
            self.df.at[idx, 'sr'] = sr
            row['audio'] = audio
            row['sr'] = sr

        # If saved bouts exist, use them without running detection.
        saved = self.current_bouts.get(idx)
        if not saved:
            existing = self.df.at[idx, 'bouts'] if 'bouts' in self.df.columns else None
            if isinstance(existing, list) and len(existing) > 0:
                saved = existing

        if saved:
            bouts = sorted(saved, key=lambda b: b['onset'])
            # Pad isn't a detection parameter, so a pad change comes through
            # with force=False and never re-runs detection. Re-derive each
            # bout's export clip boundaries (wavstart/wavend) from the current
            # pad here so pad edits take effect on saved/existing bouts — these
            # are always onset/offset ± pad, matching add_bout/update_bout.
            audio_len = (
                len(row['audio']) / row['sr']
                if isinstance(row.get('audio'), np.ndarray) and row.get('sr')
                else float('inf')
            )
            pad = params.get('pad', _DEFAULT_PARAMS['pad'])
            for b in bouts:
                b['wavstart'] = round(max(b['onset'] - pad, 0.0), 3)
                b['wavend'] = round(min(b['offset'] + pad, audio_len), 3)
            self.current_bouts[idx] = bouts
            self.df.at[idx, 'bouts'] = bouts
            return bouts, {}

        # No saved bouts — run full feature detection with the current params.
        # Re-running here on every forced recompute is what auto-finalizes the
        # parameters (the old explicit "Finalize Parameters" step). We persist
        # only the bouts list: the other feature arrays (mfcc, spectral_flux,
        # …) are not consumed anywhere in the app, and assigning a 2-D/1-D numpy
        # array into a single df cell raises a ValueError that previously
        # surfaced as a spurious "could not load recording" error.
        features = self.extractor.compute_all_features(row)
        bouts = sorted(features['bouts'], key=lambda b: b['onset'])
        self.current_bouts[idx] = bouts
        self.df.at[idx, 'bouts'] = bouts
        return bouts, features

    # ------------------------------------------------------------------
    # Bout mutations (all synchronous — no feature recompute)
    # ------------------------------------------------------------------

    def update_bout(self, idx: int, bout_id: int, onset: float, offset: float):
        """Validate and update a single bout's boundaries.

        Returns (ok, message).
        """
        bouts = self.current_bouts.get(idx, [])
        if onset >= offset:
            return False, 'Onset must be less than Offset.'
        overlaps = self._overlapping_bouts(onset, offset, bouts, exclude_idx=bout_id)
        if overlaps:
            ids = [i for i, _ in overlaps]
            return False, f'Overlaps with bout(s) {ids}. Adjust to avoid overlap.'

        row = self.df.iloc[idx]
        audio_len = (
            len(row['audio']) / row['sr']
            if isinstance(row.get('audio'), np.ndarray) and row.get('sr')
            else float('inf')
        )
        pad = self.bird_params.get(idx, _DEFAULT_PARAMS)['pad']
        wavstart = max(onset - pad, 0.0)
        wavend = min(offset + pad, audio_len)

        edited = bouts[bout_id]
        edited.update({
            'onset': round(onset, 3),
            'offset': round(offset, 3),
            'wavstart': round(wavstart, 3),
            'wavend': round(wavend, 3),
        })
        bouts.sort(key=lambda b: b['onset'])
        # Find where the edited bout landed after the sort.
        new_id = next(i for i, b in enumerate(bouts) if b is edited)
        self.df.at[idx, 'bouts'] = bouts
        self.current_bouts[idx] = bouts
        return True, f'Updated Bout {new_id}: {onset:.3f}–{offset:.3f}s', new_id

    def add_bout(self, idx: int, onset: float, offset: float):
        """Validate and append a new bout.

        Returns (ok, message).
        """
        bouts = list(self.current_bouts.get(idx, []))
        if onset >= offset:
            return False, 'Onset must be less than Offset.'
        overlaps = self._overlapping_bouts(onset, offset, bouts)
        if overlaps:
            ids = [i for i, _ in overlaps]
            return False, f'Overlaps with bout(s) {ids}.'

        row = self.df.iloc[idx]
        audio_len = (
            len(row['audio']) / row['sr']
            if isinstance(row.get('audio'), np.ndarray) and row.get('sr')
            else float('inf')
        )
        pad = self.bird_params.get(idx, _DEFAULT_PARAMS)['pad']
        new_bout = {
            'onset': round(onset, 3),
            'offset': round(offset, 3),
            'wavstart': round(max(0.0, onset - pad), 3),
            'wavend': round(min(audio_len, offset + pad), 3),
            'outlier_flag': 0,
        }
        bouts.append(new_bout)
        bouts.sort(key=lambda b: b['onset'])
        self.df.at[idx, 'bouts'] = bouts
        self.current_bouts[idx] = bouts
        return True, f'Added bout: {onset:.3f}–{offset:.3f}s'

    def remove_bouts(self, idx: int, ids):
        """Remove bouts at the given index set.

        Returns (ok, message).
        """
        bouts = self.current_bouts.get(idx, [])
        to_remove = set(ids)
        new_bouts = [b for i, b in enumerate(bouts) if i not in to_remove]
        self.df.at[idx, 'bouts'] = new_bouts
        self.current_bouts[idx] = new_bouts
        return True, f'Removed bouts: {sorted(to_remove)}'

    def set_not_outlier(self, idx: int, ids):
        """Clear outlier_flag for bouts at the given index set.

        Returns (ok, message).
        """
        bouts = self.current_bouts.get(idx, [])
        changed = []
        for i in ids:
            if i < len(bouts):
                bouts[i]['outlier_flag'] = 0
                changed.append(i)
        self.df.at[idx, 'bouts'] = bouts
        self.current_bouts[idx] = bouts
        if changed:
            return True, f'Marked as not outlier: {changed}'
        return False, 'No bouts selected.'

    # ------------------------------------------------------------------
    # Export (run on a background thread)
    # ------------------------------------------------------------------

    def export(self, idx: int, output_dir: str = 'bouts_audio'):
        """Write bout audio clips, upsert to DuckDB, regenerate CSV.

        Returns (ok, message).
        """
        import soundfile as sf

        row = self.df.iloc[idx]
        bouts = sorted(self.current_bouts.get(idx, []), key=lambda b: b['onset'])
        if not bouts:
            return False, 'No bouts to export.'

        # Crop from the original, unfiltered recording so exported clips are a
        # faithful copy of the source audio — NOT the band-pass-filtered /
        # amplitude-gated / normalized signal in row['audio'] (which is only
        # used for detection and the spectrogram). wavstart/wavend are in
        # seconds, so they slice correctly at the file's native sample rate.
        chunk_start = float(row.get('chunk_start') or 0.0)
        chunk_end = row.get('chunk_end')
        chunk_dur = float(chunk_end - chunk_start) if chunk_end is not None else None
        audio, sr = self.extractor.load_audio_raw(
            row['wav_location'], offset=chunk_start, duration=chunk_dur,
        )
        sr = int(sr)
        os.makedirs(output_dir, exist_ok=True)

        bout_rows = []
        prev_offset = None
        for bout_id, bout in enumerate(bouts):
            onset = bout['onset']
            offset = bout['offset']
            wavstart = bout['wavstart']
            wavend = bout['wavend']
            intersong = (onset - prev_offset) if prev_offset is not None else None
            duration = offset - onset
            bout_audio = audio[int(wavstart * sr): int(wavend * sr)]
            fname = f"{row['species']}_{row['bird_id']}_bout{bout_id}.wav"
            fpath = os.path.join(output_dir, fname)
            sf.write(fpath, bout_audio, sr)
            bout_rows.append({
                'species': row['species'],
                'bird_id': row['bird_id'],
                'wav_location': row['wav_location'],
                'song_id': row['song_id'],
                'bout_id': bout_id,
                'duration': duration,
                'onset': onset,
                'offset': offset,
                'wavstart': wavstart,
                'wavend': wavend,
                'intersong_interval': intersong,
                'bout_wav': fpath,
            })
            prev_offset = offset

        self.store.upsert_bouts(bout_rows)
        self.store.export_bouts_csv()
        self.exported_idxs.add(idx)
        label = f"{row['species']} {row['bird_id']}"
        return True, f'Exported {len(bout_rows)} bouts for {label}.'

    # ------------------------------------------------------------------
    # Spectrogram cache (delegates to store)
    # ------------------------------------------------------------------

    def get_cached_spectrogram(self, wav_location, audio, sr, chunk_start=0.0):
        """Return (S_db, sr) from cache or compute+store on miss."""
        import librosa

        # Include chunk_start in cache key so different chunks of the same file
        # don't collide.
        keyed_path = f"{wav_location}@{chunk_start}" if chunk_start else wav_location
        cache_key = ChatterStore.make_cache_key(
            keyed_path, sr, self.extractor.hop_length, self.extractor.frame_length,
            self.extractor.highpass_cutoff, self.extractor.lowpass_cutoff,
        )
        cached = self.store.get_cached_spectrogram(cache_key)
        if cached is not None:
            return cached
        S = librosa.stft(audio, n_fft=self.extractor.frame_length,
                         hop_length=self.extractor.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        self.store.set_cached_spectrogram(cache_key, S_db, sr)
        return S_db, sr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _overlapping_bouts(self, onset, offset, bouts, exclude_idx=None):
        return [
            (i, b) for i, b in enumerate(bouts)
            if i != exclude_idx and onset < b['offset'] and b['onset'] < offset
        ]

    def close(self):
        self.store.close()
