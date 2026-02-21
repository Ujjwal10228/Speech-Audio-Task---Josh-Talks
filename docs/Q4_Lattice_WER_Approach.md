# Question 4: Lattice-Based WER – Approach

## Objective

Given one **human reference** (which may contain errors) and **five ASR model outputs** for the same audio:

- Construct a **lattice** of valid transcription alternatives from the model outputs.
- Handle **insertions, deletions, and substitutions** so that models are not unfairly penalised when the reference is wrong.
- **Decide when to trust model agreement over the reference** (majority vote).
- Choose and justify the **alignment unit** (word / subword / phrase).
- Compute **WER** for each model using both the **original reference** and a **lattice-corrected reference**, and report both.

Goal: **Reduce WER for models that were unfairly penalised by reference errors; keep WER unchanged when the reference is correct.**

---

## Alignment Unit: Word

We use **word-level** alignment (not subword or phrase).

- **Justification**
  - WER is defined over words; word-level WER is standard and directly interpretable.
  - Subword alignment would fragment morphologically rich Hindi and inflate edit counts.
  - Phrase-level is too coarse and would miss insertions/deletions within phrases.

---

## Theory and Design

### Standard WER (unfair when reference is wrong)

- For each model: `WER(hyp, ref) = edit_distance(hyp, ref) / |ref|`.
- If the reference has an error and a model output is correct, that model is penalised (substitution/deletion/insertion). We want to correct the reference using model agreement and then recompute WER.

### Lattice and corrected reference

- **Lattice:** A structure that captures, at each time step (alignment column), which words each model produced. We build it by aligning every model hypothesis to the **reference** with a **Levenshtein (edit-distance) DP alignment**.
- **Corrected reference:** At each position, we take a **majority vote** over the words proposed by the five models (and the reference). If a fraction ≥ threshold (e.g. 0.5) of models agree on a word **different from the reference**, we replace the reference word with that consensus word at that position. Otherwise we keep the reference word.
- **Lattice WER:** For each model we compute `WER(model_output, corrected_reference)`. If the reference was correct, the corrected reference equals the original → lattice WER = standard WER. If the reference was wrong and most models agreed on the right word, the corrected reference fixes that position → models that said the right word are no longer penalised there.

### Handling insertions, deletions, substitutions

- The **alignment** step explicitly produces (ref_word, hyp_word) pairs, including empty string for insertions/deletions. Edit operations (S, D, I) are counted from this alignment.
- In the **lattice**, we work on **columns** of this alignment (one column per reference position). Majority vote is applied per column. So:
  - **Substitutions** in the reference are corrected when most models agree on another word.
  - **Insertions** in the reference (extra word) appear as an extra column; models that don’t have that word get a “deletion” in standard WER but in the lattice we can have that column voted away if most models omit it.
  - **Deletions** in the reference (missing word) appear as a column where the reference is empty; model agreement can fill in the missing word in the corrected reference.

So insertions, deletions, and substitutions in the reference are handled by the same alignment + per-column majority vote; we do not unfairly penalise models when the reference is wrong.

### When to trust model agreement over the reference

- We trust model agreement when **at least a given fraction** (e.g. 50%) of models (plus reference counted as one “vote”) agree on a word **different from the reference** at that position. Then we set the corrected reference at that position to that agreed word. Code: `majority_threshold` in `q4_lattice_wer/lattice.py` and `wer_calculator.py`.

---

## Pseudocode

```
1. ref_tokens = reference.split()
2. For each model_id, hyp: hyp_tokens[model_id] = hyp.split()
3. For each model_id: align[model_id] = Levenshtein_Alignment(ref_tokens, hyp_tokens[model_id])
4. Pad all alignments to same length (max_len) with ('','') pairs
5. For each column index c in 0..max_len-1:
   - ref_word[c] = reference word at c (from any alignment)
   - For each model_id: collect hyp_word[c] from align[model_id]
6. Corrected_ref[c] = ref_word[c]
   - votes = count of each word at column c (over models + reference)
   - If any word w != ref_word[c] has votes[w] / n_models >= threshold:
        Corrected_ref[c] = that word w
7. For each model_id:
   - WER_standard = edit_distance(hyp_tokens[model_id], ref_tokens) / len(ref_tokens)
   - WER_lattice  = edit_distance(hyp_tokens[model_id], corrected_ref) / len(corrected_ref)
8. Output table: model_id | WER_before_% | WER_after_lattice_%
```

---

## Implementation and Output

- **Code:** `q4_lattice_wer/lattice.py` (alignment, lattice merge, majority vote), `q4_lattice_wer/wer_calculator.py` (WER computation, demo, CSV output).
- **Deliverable CSV:** `data/lattice_wer_deliverable.csv` with columns e.g. `model` | `WER_before_%` | `WER_after_lattice_%`.
- **Run:** `python3 main.py --q4-demo` (synthetic example) or `python3 main.py --q4-csv [path]` with a CSV that has reference + five model output columns.

This design reduces WER for models that were unfairly penalised by reference errors and keeps it unchanged when the reference is correct.
