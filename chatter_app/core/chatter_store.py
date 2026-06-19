"""Data-access layer for Chatter.

Owns two DuckDB connections with deliberately different lifetimes:

* **Persistent** (``chatter.duckdb`` on disk): the ``bouts`` table. Survives
  restarts and backs ``chatter.bouts_df`` / the ``bouts.csv`` export.
* **Ephemeral** (``:memory:``): the ``spectrogram_cache`` table. Created fresh
  on every launch, never written to disk, and freed when the process exits or
  :meth:`ChatterStore.close` is called. Bounded by a max-entries LRU cap.

chatter_core.py talks only to this wrapper; no raw SQL lives in the UI module.
"""

import gzip
import hashlib
import io
import os
from collections import OrderedDict
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# Column order for the persistent bouts table — must match what the notebook
# expects from chatter.bouts_df today.
BOUT_COLUMNS = [
    "species", "bird_id", "wav_location", "song_id", "bout_id",
    "duration", "onset", "offset", "wavstart", "wavend",
    "intersong_interval", "bout_wav",
]

# String-typed columns, cast explicitly during CSV migration so values like a
# numeric-looking bird_id come back as strings (matching self.df).
_STRING_COLUMNS = {"species", "bird_id", "wav_location", "bout_wav"}


class ChatterStore:
    def __init__(self, duckdb_path="chatter.duckdb", csv_path="bouts.csv", spectro_cache_cap=8):
        self.duckdb_path = duckdb_path
        self.csv_path = csv_path
        self.spectro_cache_cap = spectro_cache_cap

        # --- Persistent connection (bouts) ---
        self._bouts_con = duckdb.connect(duckdb_path)
        self._bouts_con.execute(
            """
            CREATE TABLE IF NOT EXISTS bouts (
                species VARCHAR,
                bird_id VARCHAR,
                wav_location VARCHAR,
                song_id BIGINT,
                bout_id BIGINT,
                duration DOUBLE,
                onset DOUBLE,
                "offset" DOUBLE,
                wavstart DOUBLE,
                wavend DOUBLE,
                intersong_interval DOUBLE,
                bout_wav VARCHAR,
                PRIMARY KEY (species, bird_id, song_id, bout_id)
            )
            """
        )

        # --- Ephemeral connection (spectrogram cache), always empty on launch ---
        self._cache_con = duckdb.connect(":memory:")
        self._cache_con.execute(
            """
            CREATE TABLE spectrogram_cache (
                cache_key VARCHAR PRIMARY KEY,
                sr INTEGER,
                s_db_blob BLOB
            )
            """
        )
        # Tracks recency of cache keys for LRU eviction (oldest first).
        self._lru: "OrderedDict[str, None]" = OrderedDict()

        # One-time import of a pre-existing CSV when the DB starts empty.
        self._migrate_csv_if_needed()

    # ------------------------------------------------------------------ #
    # Spectrogram cache (ephemeral)
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_cache_key(wav_location, sr, hop_length, frame_length,
                       highpass_cutoff=None, lowpass_cutoff=None) -> str:
        """Stable hash of the inputs that determine the displayed spectrogram.

        The band-pass cutoffs are part of the key because the spectrogram is
        rendered from the filtered audio — changing a cutoff must yield a fresh
        spectrogram rather than a stale cache hit.
        """
        raw = (
            f"{wav_location}|{sr}|{hop_length}|{frame_length}"
            f"|{highpass_cutoff}|{lowpass_cutoff}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_spectrogram(self, cache_key) -> Optional[tuple]:
        """Return (S_db, sr) from the in-memory cache, or None on miss."""
        row = self._cache_con.execute(
            "SELECT s_db_blob, sr FROM spectrogram_cache WHERE cache_key = ?",
            [cache_key],
        ).fetchone()
        if row is None:
            return None
        blob, sr = row
        buf = io.BytesIO(gzip.decompress(bytes(blob)))
        S_db = np.load(buf, allow_pickle=False)
        self._lru.move_to_end(cache_key)  # mark most-recently-used
        return S_db, sr

    def set_cached_spectrogram(self, cache_key, S_db, sr) -> None:
        """Store a spectrogram in the in-memory cache, enforcing the LRU cap."""
        buf = io.BytesIO()
        np.save(buf, np.asarray(S_db), allow_pickle=False)
        blob = gzip.compress(buf.getvalue())

        # Upsert (delete-then-insert keeps it simple and PK-safe).
        self._cache_con.execute(
            "DELETE FROM spectrogram_cache WHERE cache_key = ?", [cache_key]
        )
        self._cache_con.execute(
            "INSERT INTO spectrogram_cache VALUES (?, ?, ?)",
            [cache_key, int(sr), blob],
        )
        self._lru[cache_key] = None
        self._lru.move_to_end(cache_key)

        # Evict least-recently-used entries beyond the cap.
        while len(self._lru) > self.spectro_cache_cap:
            old_key, _ = self._lru.popitem(last=False)
            self._cache_con.execute(
                "DELETE FROM spectrogram_cache WHERE cache_key = ?", [old_key]
            )

    # ------------------------------------------------------------------ #
    # Bouts (persistent)
    # ------------------------------------------------------------------ #
    def upsert_bouts(self, bout_rows: list) -> None:
        """Replace all stored bouts for each (species, bird_id, song_id) present
        in ``bout_rows`` with the supplied rows."""
        if not bout_rows:
            return
        new_df = pd.DataFrame(bout_rows)[BOUT_COLUMNS]

        # Replace per recording so removed bouts don't linger.
        keys = new_df[["species", "bird_id", "song_id"]].drop_duplicates()
        for _, k in keys.iterrows():
            self._bouts_con.execute(
                "DELETE FROM bouts WHERE species = ? AND bird_id = ? AND song_id = ?",
                [str(k["species"]), str(k["bird_id"]), int(k["song_id"])],
            )

        cols = ", ".join(f'"{c}"' for c in BOUT_COLUMNS)
        self._bouts_con.register("new_bouts", new_df)
        self._bouts_con.execute(
            f"INSERT INTO bouts ({cols}) SELECT {cols} FROM new_bouts"
        )
        self._bouts_con.unregister("new_bouts")

    def get_bouts_df(self) -> pd.DataFrame:
        """Return all stored bouts as a pandas DataFrame (drop-in for the old
        ``self.bouts_df``)."""
        cols = ", ".join(f'"{c}"' for c in BOUT_COLUMNS)
        df = self._bouts_con.execute(
            f'SELECT {cols} FROM bouts ORDER BY species, bird_id, song_id, onset'
        ).df()
        if df.empty:
            df = pd.DataFrame(columns=BOUT_COLUMNS)
        return df

    def get_bouts_csv_df(self) -> pd.DataFrame:
        """Read bouts.csv from disk and return it as a DataFrame.
        Returns an empty DataFrame with the correct columns if the file doesn't exist yet."""
        if self.csv_path and os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=BOUT_COLUMNS)

    def export_bouts_csv(self, path=None) -> None:
        """Write the persistent bouts table to CSV (backward-compatible export)."""
        path = path or self.csv_path
        self.get_bouts_df().to_csv(path, index=False)

    def _migrate_csv_if_needed(self) -> None:
        """If the bouts table is empty and the recording's ``<recname>.csv``
        exists, import it once so users resuming from CSV lose nothing."""
        count = self._bouts_con.execute("SELECT COUNT(*) FROM bouts").fetchone()[0]
        if count > 0 or not self.csv_path or not os.path.exists(self.csv_path):
            return
        select_exprs = []
        for c in BOUT_COLUMNS:
            if c in _STRING_COLUMNS:
                select_exprs.append(f'CAST("{c}" AS VARCHAR) AS "{c}"')
            else:
                select_exprs.append(f'"{c}"')
        cols = ", ".join(f'"{c}"' for c in BOUT_COLUMNS)
        try:
            self._bouts_con.execute(
                f"INSERT INTO bouts ({cols}) "
                f"SELECT {', '.join(select_exprs)} "
                f"FROM read_csv_auto(?, header=True)",
                [self.csv_path],
            )
        except Exception as e:
            print(f"Warning: could not import existing '{self.csv_path}' into DuckDB: {e}")

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close both connections. For the in-memory cache this frees all
        cached spectrograms immediately."""
        for con in (getattr(self, "_cache_con", None), getattr(self, "_bouts_con", None)):
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        self._lru = OrderedDict()
