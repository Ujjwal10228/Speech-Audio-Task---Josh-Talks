"""
Compute lattice-aware WER for each ASR model.

Standard WER vs Lattice WER
────────────────────────────
Standard WER:
    WER(model) = edit_distance(hyp, reference) / len(reference)

Lattice WER:
    WER(model) = edit_distance(hyp, corrected_reference) / len(corrected_reference)

Where corrected_reference is derived from majority vote across all models
(see lattice.py).  If the reference was correct, majority vote won't change it,
so the lattice WER equals the standard WER.  If the reference was wrong, models
that were right no longer get penalised.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config import DATA_DIR
from q4_lattice_wer.lattice import build_lattice, _levenshtein_alignment


def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) counts via DP alignment."""
    alignment = _levenshtein_alignment(ref, hyp)
    S = D = I = 0
    for r, h in alignment:
        if r and h and r != h:
            S += 1
        elif r and not h:
            D += 1
        elif h and not r:
            I += 1
    return S, D, I


def compute_wer_standard(hypothesis: str, reference: str) -> dict:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    S, D, I = _edit_ops(ref_tok, hyp_tok)
    N = len(ref_tok)
    wer = (S + D + I) / N if N else 0.0
    return {"S": S, "D": D, "I": I, "N": N, "wer": wer}


def compute_all_wer(
    reference: str,
    model_outputs: dict[str, str],
    majority_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute standard and lattice-corrected WER for each model.

    Returns a DataFrame with one row per model.
    """
    lattice = build_lattice(reference, model_outputs, majority_threshold)
    corrected_ref_str = " ".join(w for w in lattice.corrected_ref if w)

    rows = []
    for mid, hyp in model_outputs.items():
        std = compute_wer_standard(hyp, reference)
        lat = compute_wer_standard(hyp, corrected_ref_str)
        rows.append({
            "model":         mid,
            "std_S":         std["S"],
            "std_D":         std["D"],
            "std_I":         std["I"],
            "std_N":         std["N"],
            "std_WER_%":     round(std["wer"] * 100, 2),
            "lattice_S":     lat["S"],
            "lattice_D":     lat["D"],
            "lattice_I":     lat["I"],
            "lattice_N":     lat["N"],
            "lattice_WER_%": round(lat["wer"] * 100, 2),
            "WER_delta_%":   round((lat["wer"] - std["wer"]) * 100, 2),
        })

    df = pd.DataFrame(rows)
    return df


def demo():
    """
    Demonstrate lattice WER on a synthetic 5-model example where the
    reference contains one error ("apple" should be "orange").
    """
    # Ground-truth: "I ate an orange yesterday"
    # Reference (with error): "I ate an apple yesterday"
    reference = "I ate an apple yesterday"

    model_outputs = {
        "Model-A": "I ate an orange yesterday",     # correct; penalised by bad ref
        "Model-B": "I ate an orange yesterday",     # correct
        "Model-C": "I ate an orange yesterday",     # correct
        "Model-D": "I ate an apple yesterday",      # matches wrong reference
        "Model-E": "I ate a orange yesterday",      # minor article error
    }

    print("=" * 60)
    print("Reference (with error):", reference)
    print()

    lattice = build_lattice(reference, model_outputs)
    print("Corrected reference  :", " ".join(w for w in lattice.corrected_ref if w))
    print()

    result = compute_all_wer(reference, model_outputs)
    print(result.to_string(index=False))
    out_csv = DATA_DIR / "lattice_wer_results.csv"
    result.to_csv(out_csv, index=False)
    deliverable = result[["model", "std_WER_%", "lattice_WER_%"]].rename(
        columns={"std_WER_%": "WER_before_%", "lattice_WER_%": "WER_after_lattice_%"}
    )
    deliverable.to_csv(DATA_DIR / "lattice_wer_deliverable.csv", index=False)
    print(f"\nSaved → {out_csv}")
    print(f"Saved (PS format) → {DATA_DIR / 'lattice_wer_deliverable.csv'}")
    return result


def run_from_csv(csv_path: str, out_csv: str | None = None):
    """
    Run lattice WER on data from CSV (e.g. exported from Q4 Google Sheet).
    CSV must have: one column for reference (named 'reference' or first column),
    and columns for each model output (remaining columns or named Model-A, Model-B, ...).
    Saves PS deliverable: WER per model before and after lattice correction.
    """
    from pathlib import Path
    df = pd.read_csv(Path(csv_path), encoding="utf-8")
    # Assume first column is reference, rest are model outputs
    cols = [c for c in df.columns if str(c).strip()]
    if not cols:
        raise ValueError("CSV has no columns")
    ref_col = "reference" if "reference" in df.columns else cols[0]
    model_cols = [c for c in cols if c != ref_col]
    if not model_cols:
        raise ValueError("CSV must have at least one model column besides reference")

    # Use first row or concatenate if multiple rows (one per utterance)
    reference = str(df[ref_col].iloc[0]).strip() if len(df) == 1 else " ".join(df[ref_col].astype(str).str.strip())
    model_outputs = {}
    for c in model_cols:
        if len(df) == 1:
            model_outputs[c] = str(df[c].iloc[0]).strip()
        else:
            model_outputs[c] = " ".join(df[c].astype(str).str.strip())

    print("Reference:", reference[:80] + "..." if len(reference) > 80 else reference)
    print("Models:", list(model_outputs.keys()))

    result = compute_all_wer(reference, model_outputs)
    # PS deliverable: WER per model (before and after lattice correction)
    deliverable = result[["model", "std_WER_%", "lattice_WER_%"]].rename(
        columns={"std_WER_%": "WER_before_%", "lattice_WER_%": "WER_after_lattice_%"}
    )
    out_path = Path(out_csv) if out_csv else DATA_DIR / "lattice_wer_results.csv"
    deliverable.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved → {out_path}")
    return result


if __name__ == "__main__":
    demo()
