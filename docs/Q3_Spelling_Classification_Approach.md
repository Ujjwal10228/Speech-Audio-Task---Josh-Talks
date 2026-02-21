# Question 3: Spelling Classification – Approach

## Goal

From ~1,75,000 unique Hindi words (human-transcribed subset), label each as **correct spelling** or **incorrect spelling**, so only segments with incorrect words need re-transcription. Per guidelines: **English words transcribed in Devanagari (e.g. कंप्यूटर) count as correct.**

---

## Approach

Layered, rule- and heuristic-based classifier (no external spell-check API):

1. **Layer 1 – Script validity**  
   Reject tokens with characters outside Devanagari block (U+0900–U+097F), ZWJ/ZWNJ, extended. Mixed-script or invalid → incorrect.

2. **Layer 2 – Dictionary lookup**  
   Known-good Hindi word list (e.g. Indic NLP Library). Match → strong evidence for correct. English-in-Devanagari not rejected.

3. **Layer 3 – Morphological / phonotactic**  
   Word must start with consonant or independent vowel; no double or trailing virama. Violations → incorrect.

4. **Layer 4 – Frequency heuristic**  
   If frequency list available: very low-frequency words failing 1–3 → incorrect; high-frequency words passing script → can be correct (corpus-derived trust).

Words passing applicable layers → **correct spelling**; others → **incorrect spelling**. Code: `q3_spelling/spell_checker.py`.

---

## Balance of accuracy and efficiency

No API calls; fast over ~1.75L words. Conservative on unknown words. English-in-Devanagari not marked as errors per guidelines.

---

## Deliverables

- **a) Final number of unique correct spelled words**  
  Printed at end of run; also in output CSV (`data/spelling_classification.csv` or SPELL_OUTPUT_CSV in config).

- **b) Sheet with two columns**  
  **Word** | **Correct spelling / Incorrect spelling**  
  Same as output CSV; can be imported into Google Sheet.

Run: `python3 main.py --q3` (uses `data/word_list.csv` if present, else `data/unique_words.txt`).
