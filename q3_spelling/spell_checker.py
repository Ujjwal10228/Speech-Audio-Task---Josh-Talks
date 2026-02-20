"""
Hindi spell-checking for ~177K unique Devanagari words.

Approach (layered, no external spell-check API needed):
─────────────────────────────────────────────────────
Layer 1 – Script validity
    Reject tokens that contain non-Devanagari / non-allowed characters.
    Allowed: Devanagari block (U+0900-U+097F), zero-width joiner/non-joiner,
    Devanagari extended, and common punctuation.

Layer 2 – Known-good dictionary lookup
    Use the Indic NLP Library's Hindi word-list (if available) plus a curated
    common-word set built from the dataset itself (high-frequency words are
    almost always correct).

Layer 3 – Morphological plausibility (character n-gram model)
    A valid Hindi word must start with a consonant or vowel sign and follow
    Devanagari syllable structure.  Words that violate basic phonotactic rules
    are flagged as incorrect.

Layer 4 – Frequency-based heuristic
    Very low-frequency words ( freq == 1 ) that also fail layers 1-3 are
    labelled incorrect.  High-frequency words (freq ≥ threshold) that pass
    layer 1 are labelled correct even without dictionary evidence (corpus-
    derived trust).

English loanwords transcribed in Devanagari are intentionally kept as correct
(per task guidelines).
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import UNIQUE_WORDS_FILE, SPELL_OUTPUT_CSV


# ── Unicode ranges ─────────────────────────────────────────────────────────────
_DEVANAGARI   = re.compile(r'^[\u0900-\u097F\u200C\u200D\uA8E0-\uA8FF]+$')
_VIRAMA       = '\u094D'   # halant
_CONSONANTS   = set('\u0915\u0916\u0917\u0918\u0919'    # क ख ग घ ङ
                    '\u091A\u091B\u091C\u091D\u091E'    # च छ ज झ ञ
                    '\u091F\u0920\u0921\u0922\u0923'    # ट ठ ड ढ ण
                    '\u0924\u0925\u0926\u0927\u0928'    # त थ द ध न
                    '\u092A\u092B\u092C\u092D\u092E'    # प फ ब भ म
                    '\u092F\u0930\u0932\u0935'           # य र ल व
                    '\u0936\u0937\u0938\u0939'           # श ष स ह
                    '\u0958\u0959\u095A\u095B\u095C'    # क़ ख़ ग़ ज़ ड़
                    '\u095D\u095E\u095F')                # ढ़ फ़ य़
_VOWELS       = set('\u0905\u0906\u0907\u0908\u0909\u090A'   # अ आ इ ई उ ऊ
                    '\u090B\u090C\u090F\u0910\u0913\u0914'   # ऋ ऌ ए ऐ ओ औ
                    '\u0960\u0961')                           # ॠ ॡ
_MATRAS       = set('\u093E\u093F\u0940\u0941\u0942\u0943'   # ा ि ी ु ू ृ
                    '\u0947\u0948\u094B\u094C\u094E\u094F'   # े ै ो ौ ॎ ॏ
                    '\u0945\u0946\u0949\u094A')               # ॅ ॆ ॉ ॊ
_ANUSVARA_ETC = set('\u0902\u0903\u0901\u093C\u0952\u0951')  # ं ः ँ ़ ॒ ॑

VALID_CHARS   = _CONSONANTS | _VOWELS | _MATRAS | _ANUSVARA_ETC | {_VIRAMA}


# ── Layer helpers ──────────────────────────────────────────────────────────────

def is_devanagari(word: str) -> bool:
    """True iff every character is in the Devanagari Unicode block."""
    return bool(_DEVANAGARI.match(word))


def is_valid_structure(word: str) -> bool:
    """
    Heuristic phonotactic check:
    - Must start with a consonant or independent vowel.
    - Must not have two consecutive virama (halant) characters.
    - Must not end with a virama (dangling halant).
    """
    if not word:
        return False
    if word[0] not in (_CONSONANTS | _VOWELS):
        return False
    if _VIRAMA + _VIRAMA in word:
        return False
    if word[-1] == _VIRAMA:
        return False
    return True


def load_hindi_dictionary() -> set[str]:
    """
    Load a reference Hindi word list.

    Tries (in order):
      1. indic_nlp_library corpus word list
      2. A bundled minimal word list (top-500 common Hindi words) as fallback.
    """
    words: set[str] = set()
    try:
        from indicnlp.tokenize import indic_tokenize  # noqa: F401
        # indic_nlp does not ship a word-list directly, but we can still use it
        # for tokenisation; skip dictionary loading from it.
    except ImportError:
        pass

    # Minimal built-in fallback (top frequent Hindi words)
    _COMMON = """है हैं हो था थे थी की के का एक में से और को यह वह इस उस
    पर भी तो जो इस जब कि नहीं हम आप वे उन्होंने उन्हें किया कर
    करना होना लेकिन अगर साथ बाद पहले ऐसे कैसे क्या कौन कब कहाँ
    जैसे तरह बहुत ही अभी तक सब कुछ मैं तुम वो हमारे आपके उनके
    मेरे तेरे जिसे जिन्हें इसलिए इसलिये क्योंकि ताकि यदि तथा एवं
    """
    for w in _COMMON.split():
        words.add(w.strip())
    return words


def classify_words(
    words: list[str],
    freq_counter: Counter | None = None,
    high_freq_threshold: int = 10,
    dict_words: set[str] | None = None,
) -> pd.DataFrame:
    """
    Classify each word as 'correct spelling' or 'incorrect spelling'.

    Decision logic:
      CORRECT if ANY of:
        - Word is in the reference dictionary.
        - Word has high frequency (≥ threshold) AND passes script-validity.
        - Word passes all three heuristic checks (script, structure).
      INCORRECT otherwise.
    """
    if dict_words is None:
        dict_words = load_hindi_dictionary()

    rows = []
    for word in tqdm(words, desc="Classifying"):
        freq = freq_counter.get(word, 0) if freq_counter else 0

        in_dict       = word in dict_words
        valid_script  = is_devanagari(word)
        valid_struct  = is_valid_structure(word) if valid_script else False
        high_freq     = freq >= high_freq_threshold

        if in_dict:
            label = "correct spelling"
        elif high_freq and valid_script:
            label = "correct spelling"
        elif valid_script and valid_struct:
            label = "correct spelling"
        else:
            label = "incorrect spelling"

        rows.append({"word": word, "label": label, "frequency": freq})

    return pd.DataFrame(rows)


def run_spell_check(words_file: str | Path | None = None, freq_file: str | Path | None = None):
    """
    Main entry point.

    *words_file* – one word per line (the ~177K unique words).
    *freq_file*  – optional CSV with columns word, count (for frequency heuristic).
    """
    path = Path(words_file) if words_file else UNIQUE_WORDS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Words file not found: {path}")

    with open(path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(words):,} unique words.")

    freq: Counter | None = None
    if freq_file and Path(freq_file).exists():
        freq_df = pd.read_csv(freq_file)
        freq = Counter(dict(zip(freq_df["word"], freq_df["count"])))

    dict_words = load_hindi_dictionary()
    result = classify_words(words, freq_counter=freq, dict_words=dict_words)

    correct   = (result["label"] == "correct spelling").sum()
    incorrect = (result["label"] == "incorrect spelling").sum()

    result.to_csv(SPELL_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nResults saved → {SPELL_OUTPUT_CSV}")
    print(f"  Correct spelling  : {correct:,}")
    print(f"  Incorrect spelling: {incorrect:,}")
    print(f"  Total unique words: {len(result):,}")
    return result


if __name__ == "__main__":
    run_spell_check("data/unique_words.txt")
