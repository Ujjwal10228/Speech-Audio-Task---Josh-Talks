# Question 1: Data Preprocessing, Fine-tuning & WER Report

## a) Data Preprocessing – What We Did

**Data access**
- Index CSV columns used: `user_id`, `recording_id`, `language`, `duration`, `rec_url_gcp`, `transcription_url_gcp`, `metadata_url_gcp` (or `transcription_url` / `metadata_url`). When these URL columns are present, they are used as-is so alternative GCP bases (e.g. `upload_goai`, `joshtalks-data-collection`) work without code change.
- Downloaded from GCP: raw WAV (audio), JSON (transcription with segment-level `text`, `start_time`, `end_time`), JSON (metadata). All saved under `data/audio/`, `data/transcriptions/`, `data/metadata/`.

**Filtering**
- Only Hindi rows: `language == "hi"`.
- Segment duration: drop segments shorter than 0.5 s or longer than 30 s (Whisper’s limit).

**Text cleaning (per segment)**
- Unicode NFC normalisation.
- Remove ASCII punctuation and digits; collapse whitespace.
- Devanagari transcriptions and English-in-Devanagari words kept as per guidelines.

**Audio**
- Load with librosa at 16 kHz mono; peak-normalise to [-1, 1]; validate duration; write 16-bit PCM WAV. Segments outside the duration bounds are excluded from the manifest.

**Training manifest**
- Built from index + transcription JSONs: one row per segment with `audio_path`, `text`, `duration` (and `start_time`/`end_time`). Output: `data/train_manifest.csv`. This is the input for Whisper fine-tuning.

---

## b) Fine-tuning and Evaluation

- **Model:** Whisper-small (`openai/whisper-small`), fine-tuned on the Hindi segments from the 10-hour dataset.
- **Evaluation:** Both the **pretrained Whisper-small baseline** and the **fine-tuned model** are evaluated on the **Hindi portion of the FLEURS test set**.
- Training and evaluation scripts: `q1_whisper_finetune/train.py`, `evaluate.py`. Entry point: `main.py --q1` (full pipeline) or `main.py --q1-eval` (evaluation only).

---

## c) WER – Structured Table

WER is reported in a structured table. The pipeline writes:

- **File:** `data/wer_results_deliverable.csv`
- **Columns:** `Model` | `Dataset` | `WER` (or equivalent: baseline vs fine-tuned, FLEURS Hindi).

Example format:

| Model              | Dataset     | WER (%) |
|--------------------|------------|---------|
| Whisper-small (baseline) | FLEURS Hindi | …       |
| Whisper-small (fine-tuned) | FLEURS Hindi | …       |

Run `python3 main.py --q1` (or `--q1-eval` for evaluation only) to regenerate this table after training/evaluation.
