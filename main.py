"""
Josh Talks – AI Researcher Intern Assignment
============================================
Single entry point to run any or all of the four questions.

Sample usage
------------
# Run the full pipeline end-to-end (assumes data/index.csv exists)
    python main.py

# Run only Q4 demo (no data download needed)
    python main.py --q4-demo
"""
import sys
from pathlib import Path

# ── Q4 demo (self-contained, no data needed) ──────────────────────────────────

def run_q4_demo():
    from q4_lattice_wer.wer_calculator import demo
    demo()


# ── Q3 spell check ────────────────────────────────────────────────────────────

def run_q3(words_file: str = "data/unique_words.txt"):
    from q3_spelling.spell_checker import run_spell_check
    run_spell_check(words_file)


# ── Q2 disfluency pipeline ────────────────────────────────────────────────────

def run_q2(index_csv: str = "data/index.csv"):
    from q2_disfluency.pipeline import run_pipeline
    run_pipeline(index_csv)


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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--q4-demo" in args:
        print("\n── Q4: Lattice WER Demo ──")
        run_q4_demo()

    elif "--q3" in args:
        print("\n── Q3: Spelling Classification ──")
        run_q3()

    elif "--q2" in args:
        print("\n── Q2: Disfluency Detection ──")
        run_q2()

    elif "--q1" in args:
        print("\n── Q1: Whisper Fine-tuning & Evaluation ──")
        run_q1()

    else:
        print("Usage: python main.py [--q1 | --q2 | --q3 | --q4-demo]")
        print("\nRunning Q4 demo (self-contained)...")
        run_q4_demo()
