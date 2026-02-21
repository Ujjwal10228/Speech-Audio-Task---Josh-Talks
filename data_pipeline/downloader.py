"""
Download audio, transcription, and metadata files from GCP.
Reads a CSV index file with columns: user_id, recording_id, language, duration,
rec_url_gcp, transcription_url, metadata_url.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    AUDIO_DIR, TRANSCRIPTION_DIR, METADATA_DIR,
    GCP_BASE, rec_url, transcription_url, metadata_url,
)


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single URL to dest, skipping if already present."""
    if dest.exists():
        return True
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  [WARN] Failed {url}: {exc}")
                return False
            time.sleep(2 ** attempt)
    return False


def download_dataset(index_csv: str | Path, workers: int = 8, language_filter: str = "hi"):
    """
    Download all files referenced in *index_csv*.

    Uses URL columns from CSV when present (rec_url_gcp, transcription_url_gcp,
    metadata_url_gcp or transcription_url, metadata_url), so that alternative
    base URLs (e.g. joshtalks-data-collection) work. Otherwise builds URLs
    from user_id and recording_id using config GCP_BASE.
    Returns a DataFrame of successfully downloaded rows.
    """
    df = pd.read_csv(index_csv)
    if language_filter:
        df = df[df["language"] == language_filter].reset_index(drop=True)
    print(f"Rows to download: {len(df)}")

    has_url_cols = "rec_url_gcp" in df.columns and "transcription_url_gcp" in df.columns
    if not has_url_cols and "rec_url_gcp" in df.columns and "transcription_url" in df.columns:
        has_url_cols = True
    meta_col = "metadata_url_gcp" if "metadata_url_gcp" in df.columns else "metadata_url"

    tasks = []
    for _, row in df.iterrows():
        rid = str(row["recording_id"])
        if has_url_cols:
            url_audio = str(row["rec_url_gcp"]).strip()
            url_trans = str(row.get("transcription_url_gcp", row.get("transcription_url", ""))).strip()
            url_meta = str(row.get(meta_col, "")).strip()
        else:
            uid = str(row["user_id"])
            url_audio = rec_url(uid, rid)
            url_trans = transcription_url(uid, rid)
            url_meta = metadata_url(uid, rid)
        tasks.append((url_audio, AUDIO_DIR / f"{rid}.wav"))
        tasks.append((url_trans, TRANSCRIPTION_DIR / f"{rid}_transcription.json"))
        tasks.append((url_meta, METADATA_DIR / f"{rid}_metadata.json"))

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_file, url, dest): (url, dest) for url, dest in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"Download complete. OK={ok}  FAIL={fail}")
    return df


def load_transcription(recording_id: str) -> list[dict]:
    """
    Load transcription JSON for a recording.

    Returns a list of segment dicts, each with at least:
      text, start_time, end_time  (seconds).
    """
    path = TRANSCRIPTION_DIR / f"{recording_id}_transcription.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list-of-segments and nested {"segments": [...]} formats
    if isinstance(data, list):
        return data
    return data.get("segments", data.get("utterances", []))


if __name__ == "__main__":
    # Example: python -m data_pipeline.downloader
    import sys
    index = sys.argv[1] if len(sys.argv) > 1 else "data/index.csv"
    download_dataset(index)
