"""
Detect speech disfluencies in Hindi transcription segments.

Detection strategy (text-based, no additional models needed):
  1. Filler-word lookup  – match against a curated Hindi+romanised filler list.
  2. Repetition regex    – catch "word word" patterns (e.g. "मैं मैं").
  3. Prolongation regex  – catch character runs ≥ 3 (e.g. "हाँऽऽऽ", "soooo").
  4. False-start marker  – ellipsis / dash mid-word ("मैं… मतलब").

Each detected item is tagged with one or more disfluency types.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from config import HINDI_FILLERS, REPETITION_PATTERN, PROLONGATION_PATTERN


# False-start: word followed by "…" / "..." / "—" / "--"
_FALSE_START = re.compile(r'\b\w+[…—\-]{1,3}\s', re.UNICODE)

# Hesitation: isolated short syllables like "ए-", "अ-"
_HESITATION = re.compile(r'\b[अआइईउऊएओ]\b', re.UNICODE)


@dataclass
class DisfluencyHit:
    segment_index: int
    start_time: float
    end_time: float
    text: str
    types: list[str] = field(default_factory=list)
    matched_token: str = ""


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def detect_fillers(text: str) -> list[str]:
    """Return filler words found in *text*."""
    tokens = re.split(r'\s+', _normalise(text).lower())
    return [t for t in tokens if t in HINDI_FILLERS]


def detect_repetitions(text: str) -> list[str]:
    return REPETITION_PATTERN.findall(text)


def detect_prolongations(text: str) -> list[str]:
    return PROLONGATION_PATTERN.findall(text)


def detect_false_starts(text: str) -> list[str]:
    return _FALSE_START.findall(text)


def detect_hesitations(text: str) -> list[str]:
    return _HESITATION.findall(text)


def analyse_segment(seg_idx: int, segment: dict) -> list[DisfluencyHit]:
    """
    Analyse one transcription segment dict and return a list of DisfluencyHit.

    *segment* must have keys: text, start_time, end_time.
    A single segment can produce multiple hits if multiple types are detected.
    """
    text = segment.get("text", "")
    start = float(segment.get("start_time", 0))
    end   = float(segment.get("end_time", 0))

    hits: list[DisfluencyHit] = []

    # Aggregate all types found for this segment
    types_found: list[tuple[str, str]] = []  # (type_label, matched_token)

    for token in detect_fillers(text):
        types_found.append(("filler", token))
    for token in detect_repetitions(text):
        types_found.append(("repetition", token))
    for token in detect_prolongations(text):
        types_found.append(("prolongation", token))
    for token in detect_false_starts(text):
        types_found.append(("false_start", token.strip()))
    for token in detect_hesitations(text):
        types_found.append(("hesitation", token))

    if not types_found:
        return []

    # One row per unique (segment, disfluency_type)
    seen_types: set[str] = set()
    for dtype, token in types_found:
        if dtype in seen_types:
            continue
        seen_types.add(dtype)
        hits.append(DisfluencyHit(
            segment_index=seg_idx,
            start_time=start,
            end_time=end,
            text=text,
            types=[dtype],
            matched_token=token,
        ))

    return hits


def analyse_recording(segments: list[dict]) -> list[DisfluencyHit]:
    """Run disfluency detection over all segments of a recording."""
    hits = []
    for i, seg in enumerate(segments):
        hits.extend(analyse_segment(i, seg))
    return hits
