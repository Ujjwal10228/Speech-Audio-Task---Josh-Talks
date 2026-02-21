"""Central configuration for all questions."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTION_DIR = DATA_DIR / "transcriptions"
METADATA_DIR = DATA_DIR / "metadata"
CLIPPED_AUDIO_DIR = DATA_DIR / "clipped_audio"

for _d in (AUDIO_DIR, TRANSCRIPTION_DIR, METADATA_DIR, CLIPPED_AUDIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── GCP Storage base ───────────────────────────────────────────────────────────
GCP_BASE = "https://storage.googleapis.com/upload_goai"

def rec_url(user_id: str, recording_id: str) -> str:
    return f"{GCP_BASE}/{user_id}/{recording_id}.wav"

def transcription_url(user_id: str, recording_id: str) -> str:
    return f"{GCP_BASE}/{user_id}/{recording_id}_transcription.json"

def metadata_url(user_id: str, recording_id: str) -> str:
    return f"{GCP_BASE}/{user_id}/{recording_id}_metadata.json"

# ── Q1 – Whisper fine-tuning ───────────────────────────────────────────────────
WHISPER_MODEL = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"
SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION = 30       # seconds – Whisper's hard limit
MIN_AUDIO_DURATION = 0.5      # drop sub-0.5 s clips
TRAIN_OUTPUT_DIR = ROOT / "q1_whisper_finetune" / "checkpoints"
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Q2 – Disfluency detection ──────────────────────────────────────────────────
DISFLUENCY_OUTPUT_CSV = DATA_DIR / "disfluency_dataset.csv"
CLIPPED_AUDIO_BASE_URL = ""  # set if you upload clips somewhere

# Hindi disfluency word-list (Devanagari + romanised variants)
HINDI_FILLERS = {
    # single-word fillers
    "उह", "अह", "हम्म", "हम", "ओह", "ओ", "आ", "हाँ", "बस",
    "मतलब", "यानी", "यानि", "वो", "वह", "अरे", "तो", "ना",
    "सो", "लाइक", "बेसिकली", "एक्चुअली", "ओके", "ठीक",
    # romanised (may appear in transcriptions)
    "uh", "um", "umm", "hmm", "ah", "oh", "er", "erm",
}

# Regex patterns for repetitions / false starts (applied to segment text)
import re
REPETITION_PATTERN = re.compile(
    r'(?<!\S)(\S+)\s+\1(?!\S)',   # word word (Unicode-safe boundaries)
    re.UNICODE,
)
PROLONGATION_PATTERN = re.compile(
    r'(\w)\1{2,}',               # aaaa / soooo
    re.UNICODE,
)

# ── Q3 – Spell check ───────────────────────────────────────────────────────────
UNIQUE_WORDS_FILE = DATA_DIR / "unique_words.txt"
SPELL_OUTPUT_CSV = DATA_DIR / "spelling_classification.csv"

# ── Q4 – Lattice WER ──────────────────────────────────────────────────────────
MAJORITY_VOTE_THRESHOLD = 0.5   # fraction of models that must agree to override reference
