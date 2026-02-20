"""
Evaluate pretrained Whisper-small baseline and the fine-tuned model on the
FLEURS Hindi test set.  Prints and saves a WER comparison table.
"""
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate as hf_evaluate

from config import WHISPER_MODEL, LANGUAGE, TASK, SAMPLE_RATE, TRAIN_OUTPUT_DIR
from q1_whisper_finetune.dataset import load_fleurs_hindi_test
from data_pipeline.preprocessor import clean_text


wer_metric = hf_evaluate.load("wer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_dataset(model, processor, dataset, batch_size: int = 8) -> list[str]:
    """Run inference on a HuggingFace dataset, return list of predicted strings."""
    model.eval()
    predictions = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Transcribing"):
        batch = dataset[i: i + batch_size]
        arrays = [np.array(a["array"], dtype=np.float32) for a in batch["audio"]]
        inputs = processor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANGUAGE, task=TASK
                ),
            )
        decoded = processor.batch_decode(ids, skip_special_tokens=True)
        predictions.extend(decoded)
    return predictions


def compute_wer(predictions: list[str], references: list[str]) -> float:
    preds_clean = [clean_text(p) for p in predictions]
    refs_clean  = [clean_text(r) for r in references]
    return wer_metric.compute(predictions=preds_clean, references=refs_clean)


def evaluate_models(finetuned_dir: str | Path | None = None):
    """
    Evaluate baseline and (optionally) fine-tuned model.

    Returns a DataFrame with columns: model, wer.
    """
    test_ds = load_fleurs_hindi_test()
    references = test_ds["sentence"]

    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n=== Baseline: openai/whisper-small ===")
    base_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=LANGUAGE, task=TASK)
    base_model     = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(DEVICE)
    base_preds     = transcribe_dataset(base_model, base_processor, test_ds)
    base_wer       = compute_wer(base_preds, references)
    results.append({"model": "Whisper-small (baseline)", "wer": round(base_wer * 100, 2)})
    print(f"Baseline WER: {base_wer * 100:.2f}%")

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    ft_dir = Path(finetuned_dir) if finetuned_dir else TRAIN_OUTPUT_DIR / "best_model"
    if ft_dir.exists():
        print(f"\n=== Fine-tuned: {ft_dir} ===")
        ft_processor = WhisperProcessor.from_pretrained(ft_dir, language=LANGUAGE, task=TASK)
        ft_model     = WhisperForConditionalGeneration.from_pretrained(ft_dir).to(DEVICE)
        ft_preds     = transcribe_dataset(ft_model, ft_processor, test_ds)
        ft_wer       = compute_wer(ft_preds, references)
        results.append({"model": "Whisper-small (fine-tuned on JoshTalks-Hi)", "wer": round(ft_wer * 100, 2)})
        print(f"Fine-tuned WER: {ft_wer * 100:.2f}%")
    else:
        print(f"[INFO] No fine-tuned model found at {ft_dir}. Run train.py first.")

    table = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("WER Results on FLEURS Hindi Test Set")
    print("=" * 50)
    print(table.to_string(index=False))
    table.to_csv("data/wer_results.csv", index=False)
    return table


if __name__ == "__main__":
    evaluate_models()
