# Question 2: Disfluency Detection – Methodology Summary

## a) Deliverables (Short Answers)

### How we detected the disfluencies

Text-based detection on segment-level transcriptions (no extra ASR). For each segment:

1. **Filler-word lookup** – Curated Hindi + romanised fillers (e.g. उह, अह, हम्म, um, uh, hmm) matched against segment tokens.
2. **Repetition regex** – Adjacent repeated word (e.g. "word word").
3. **Prolongation regex** – Same character repeated ≥ 3 times (e.g. soooo).
4. **False-start marker** – Word followed by ellipsis/dash (…, —, --).
5. **Hesitation** – Isolated short vowels as hesitation markers.

Each segment can have multiple types; each is tagged with disfluency_type and matched_token. Code: `q2_disfluency/detector.py`, pipeline: `q2_disfluency/pipeline.py`.

### How we clipped the audio segment from the complete recording

For each detected disfluency we have recording_id, start_time, end_time from the transcription JSON. We use pydub to load the full WAV, slice [start_time, end_time] in milliseconds, and export a short WAV clip. Clips saved under `data/clipped_audio/` with names like `{recording_id}_{start_ms}_{end_ms}.wav`. Code: `q2_disfluency/audio_clipper.py`.

### Preprocessing / normalisation applied

- **Text:** Unicode NFC normalisation and strip before filler/regex checks.
- **Audio:** No extra normalisation for clipping; we use the same full recording WAVs. Clipped segments are raw slices.

---

## b) Output Dataset (Sheet Format)

- **File:** `data/disfluency_dataset.csv`
- **Schema (one row per disfluency):** recording_id | disfluency_type | start_time | end_time | segment_text | matched_token | clip_filename | segmented_audio_link

segmented_audio_link is a URL if CLIPPED_AUDIO_BASE_URL is set in config.py after uploading clips, else local path.

---

## c) Segmented Audio Files

Clips are in `data/clipped_audio/`. Each CSV row has a clip_filename pointing to one WAV. Upload these and set CLIPPED_AUDIO_BASE_URL so the sheet shows links.
