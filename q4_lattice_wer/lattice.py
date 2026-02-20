"""
Word-level Lattice construction from multiple ASR model outputs.

Design
──────
Alignment unit: **word** (not subword / phrase).
Justification:
  - WER is inherently word-level, so aligning at the word level is natural and
    directly interpretable.
  - Subword alignment would fragment multi-syllabic words, inflating edit
    distances for morphologically rich languages (Hindi).
  - Phrase-level is too coarse; intra-phrase insertions would be missed.

Lattice structure
─────────────────
A lattice is a directed acyclic graph (DAG):
  - Each node represents a position in the merged word sequence.
  - Each edge is labelled with a word alternative and a set of sources
    (model IDs that produced that word at that position).
  - The reference is one of the sources; model outputs are the others.

Majority-vote reference correction
───────────────────────────────────
At each alignment position, if ≥ threshold of models agree on a word that
differs from the reference, we replace the reference with the consensus word
for WER computation.  This prevents unfair penalisation when the reference
itself is erroneous.

Alignment algorithm
───────────────────
We use ROVER-style (Recogniser Output Voting Error Reduction) pairwise
alignment:
  1. Align each model output to the reference using dynamic programming
     (Levenshtein edit alignment).
  2. Merge aligned columns into a lattice.
  3. Apply majority vote per column to produce the "corrected reference".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Alignment ─────────────────────────────────────────────────────────────────

def _levenshtein_alignment(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """
    Return aligned (ref_word, hyp_word) pairs using classic DP edit alignment.
    Gaps are represented as '' (empty string).
    """
    R, H = len(ref), len(hyp)
    # DP table: costs
    dp = np.zeros((R + 1, H + 1), dtype=int)
    dp[:, 0] = np.arange(R + 1)
    dp[0, :] = np.arange(H + 1)

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j],     # deletion
                                   dp[i, j - 1],       # insertion
                                   dp[i - 1, j - 1])   # substitution

    # Backtrack
    alignment: list[tuple[str, str]] = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))   # substitution
            i -= 1; j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            alignment.append((ref[i - 1], ''))            # deletion
            i -= 1
        else:
            alignment.append(('', hyp[j - 1]))            # insertion
            j -= 1

    return list(reversed(alignment))


# ── Lattice node / edge ────────────────────────────────────────────────────────

@dataclass
class LatticeNode:
    idx: int


@dataclass
class LatticeEdge:
    src: int
    dst: int
    word: str
    sources: set[str] = field(default_factory=set)   # model IDs


@dataclass
class WordLattice:
    """
    Merged word lattice built from multiple ASR hypotheses aligned to a reference.
    """
    nodes: list[LatticeNode]
    edges: list[LatticeEdge]
    ref_words: list[str]          # original reference words (per column)
    corrected_ref: list[str]      # majority-vote corrected reference

    def best_path_for_model(self, model_id: str) -> list[str]:
        """Extract the hypothesis path for a specific model from the lattice."""
        path = []
        for edge in self.edges:
            if model_id in edge.sources and edge.word:
                path.append(edge.word)
        return path


# ── Lattice builder ────────────────────────────────────────────────────────────

def build_lattice(
    reference: str,
    model_outputs: dict[str, str],
    majority_threshold: float = 0.5,
) -> WordLattice:
    """
    Build a merged word lattice from *model_outputs* aligned to *reference*.

    Parameters
    ----------
    reference       : Ground-truth transcription string.
    model_outputs   : {model_id: hypothesis_string}.
    majority_threshold : Fraction of models that must agree to override reference.

    Returns a WordLattice with corrected_ref applied.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens_map = {mid: hyp.lower().split() for mid, hyp in model_outputs.items()}

    # Step 1 – Align each model to the reference
    alignments: dict[str, list[tuple[str, str]]] = {}
    for mid, hyp_tokens in hyp_tokens_map.items():
        alignments[mid] = _levenshtein_alignment(ref_tokens, hyp_tokens)

    # Step 2 – Merge into a column-based lattice
    # We extend all alignments to the same length by appending ('','') pairs
    max_len = max((len(a) for a in alignments.values()), default=0)
    padded: dict[str, list[tuple[str, str]]] = {}
    for mid, aln in alignments.items():
        padded[mid] = aln + [('', '')] * (max_len - len(aln))

    ref_col:  list[str] = []
    hyp_cols: dict[str, list[str]] = {mid: [] for mid in model_outputs}

    for col_idx in range(max_len):
        ref_word = ''
        for mid in model_outputs:
            r, h = padded[mid][col_idx]
            if r:
                ref_word = r
                break
        ref_col.append(ref_word)
        for mid in model_outputs:
            _, h = padded[mid][col_idx]
            hyp_cols[mid].append(h)

    # Step 3 – Majority vote per column
    n_models = len(model_outputs)
    corrected_ref: list[str] = []
    for col_idx in range(max_len):
        votes: dict[str, int] = {}
        for mid in model_outputs:
            w = hyp_cols[mid][col_idx]
            if w:
                votes[w] = votes.get(w, 0) + 1
        if votes:
            top_word, top_count = max(votes.items(), key=lambda x: x[1])
            if (top_count / n_models >= majority_threshold
                    and top_word != ref_col[col_idx]):
                corrected_ref.append(top_word)
            else:
                corrected_ref.append(ref_col[col_idx])
        else:
            corrected_ref.append(ref_col[col_idx])

    # Step 4 – Build DAG
    nodes = [LatticeNode(i) for i in range(max_len + 1)]
    edges: list[LatticeEdge] = []
    for col_idx in range(max_len):
        # Reference edge
        ref_edge = LatticeEdge(col_idx, col_idx + 1, ref_col[col_idx], {"reference"})
        edges.append(ref_edge)
        # Model edges (merge same-word edges)
        word_to_edge: dict[str, LatticeEdge] = {}
        for mid in model_outputs:
            w = hyp_cols[mid][col_idx]
            if w not in word_to_edge:
                word_to_edge[w] = LatticeEdge(col_idx, col_idx + 1, w, set())
            word_to_edge[w].sources.add(mid)
        edges.extend(word_to_edge.values())

    return WordLattice(
        nodes=nodes,
        edges=edges,
        ref_words=ref_col,
        corrected_ref=corrected_ref,
    )
