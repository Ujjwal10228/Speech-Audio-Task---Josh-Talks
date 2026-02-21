"""
Josh Talks – AI Researcher Intern Assignment
============================================
Single entry point to run any or all of the four questions.

Sample usage
------------
# Run the full pipeline end-to-end (assumes data/index.csv exists)
    python3 main.py --q1

# Run only Q4 demo (no data download needed)
    python3 main.py --q4-demo

# Run Q3 spell-check demo (no external word file needed)
    python3 main.py --q3-demo

# Run Q2 disfluency demo (no audio needed)
    python3 main.py --q2-demo

# Run Q1 evaluation only on FLEURS Hindi (no training, no GCP data)
    python3 main.py --q1-eval

# Run all demos at once
    python3 main.py --all-demo
"""
import sys
from pathlib import Path

# ── Q4 demo (self-contained, no data needed) ──────────────────────────────────

def run_q4_demo():
    from q4_lattice_wer.wer_calculator import demo
    demo()


def run_q4_csv(csv_path: str = "data/lattice_input.csv"):
    """Q4 from CSV (export Lattice Google Sheet as CSV: reference + 5 model columns)."""
    from q4_lattice_wer.wer_calculator import run_from_csv
    run_from_csv(csv_path)


# ── Q3 spell check ────────────────────────────────────────────────────────────

def run_q3(words_file: str = "data/unique_words.txt"):
    from q3_spelling.spell_checker import run_spell_check
    run_spell_check(words_file)


def run_q3_demo():
    from q3_spelling.spell_checker import demo
    demo()


def run_q3_csv(csv_path: str = "data/word_list.csv"):
    """Q3 from CSV (export Word List Google Sheet as CSV)."""
    from q3_spelling.spell_checker import run_spell_check_from_csv
    run_spell_check_from_csv(csv_path)


# ── Q2 disfluency pipeline ────────────────────────────────────────────────────

def run_q2(index_csv: str = "data/index.csv"):
    from q2_disfluency.pipeline import run_pipeline
    run_pipeline(index_csv)


def run_q2_demo():
    from q2_disfluency.pipeline import demo
    demo()


# ── Q1 Whisper fine-tune + evaluate ──────────────────────────────────────────

def run_q1(index_csv: str = "data/index.csv"):
    from data_pipeline.downloader import download_dataset
    from data_pipeline.preprocessor import build_training_manifest
    from q1_whisper_finetune.train import train
    from q1_whisper_finetune.evaluate import evaluate_models

    df = download_dataset(index_csv)
    build_training_manifest(df)
    train("data/train_manifest.csv")
    evaluate_models()


def run_q1_eval():
    from q1_whisper_finetune.evaluate import evaluate_models
    evaluate_models()


def run_q1_demo():
    from q1_whisper_finetune.evaluate import demo
    demo()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--all-demo" in args:
        print("\n" + "=" * 60)
        print("  Running ALL demos")
        print("=" * 60)

        print("\n── Q1: Whisper WER Demo ──")
        run_q1_demo()

        print("\n── Q2: Disfluency Detection Demo ──")
        run_q2_demo()

        print("\n── Q3: Spelling Classification Demo ──")
        run_q3_demo()

        print("\n── Q4: Lattice WER Demo ──")
        run_q4_demo()

        print("\nAll demo outputs saved to data/")

    elif "--q4-demo" in args:
        print("\n── Q4: Lattice WER Demo ──")
        run_q4_demo()

    elif "--q4-csv" in args:
        idx = args.index("--q4-csv")
        path = args[idx + 1] if idx + 1 < len(args) else "data/lattice_input.csv"
        print("\n── Q4: Lattice WER from CSV ──")
        run_q4_csv(path)

    elif "--q3-csv" in args:
        idx = args.index("--q3-csv")
        path = args[idx + 1] if idx + 1 < len(args) else "data/word_list.csv"
        print("\n── Q3: Spell check from CSV (Word List sheet) ──")
        run_q3_csv(path)

    elif "--q3-demo" in args:
        print("\n── Q3: Spelling Classification Demo ──")
        run_q3_demo()

    elif "--q3" in args:
        print("\n── Q3: Spelling Classification ──")
        run_q3()

    elif "--q2-demo" in args:
        print("\n── Q2: Disfluency Detection Demo ──")
        run_q2_demo()

    elif "--q2" in args:
        print("\n── Q2: Disfluency Detection ──")
        run_q2()

    elif "--q1-demo" in args:
        print("\n── Q1: Whisper WER Demo ──")
        run_q1_demo()

    elif "--q1-eval" in args:
        print("\n── Q1: Whisper Evaluation (baseline on FLEURS Hindi) ──")
        run_q1_eval()

    elif "--q1" in args:
        print("\n── Q1: Whisper Fine-tuning & Evaluation ──")
        run_q1()

    else:
        print("Usage: python3 main.py [OPTION]")
        print()
        print("  Demo modes (no external data needed):")
        print("    --all-demo    Run all 4 demos together")
        print("    --q1-demo     Whisper WER comparison (benchmark numbers)")
        print("    --q2-demo     Disfluency detection demo (synthetic segments)")
        print("    --q3-demo     Spelling classification demo (sample Hindi words)")
        print("    --q4-demo     Lattice WER demo (synthetic example)")
        print()
        print("  Full pipeline modes (require data/index.csv from GCP):")
        print("    --q1          Download data → fine-tune Whisper → evaluate")
        print("    --q1-eval     Evaluate baseline Whisper on FLEURS Hindi (no training)")
        print("    --q2          Disfluency detection on full dataset")
        print("    --q3          Spell check (requires data/unique_words.txt)")
        print()
        print("Running --all-demo by default...")

        print("\n── Q1: Whisper WER Demo ──")
        run_q1_demo()

        print("\n── Q2: Disfluency Detection Demo ──")
        run_q2_demo()

        print("\n── Q3: Spelling Classification Demo ──")
        run_q3_demo()

        print("\n── Q4: Lattice WER Demo ──")
        run_q4_demo()

        print("\nAll demo outputs saved to data/")
