"""
Fine-tune Whisper-small on the Josh Talks Hindi dataset.

Usage (from repo root):
    python -m q1_whisper_finetune.train
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from config import WHISPER_MODEL, LANGUAGE, TASK, TRAIN_OUTPUT_DIR, SAMPLE_RATE
from q1_whisper_finetune.dataset import build_hf_dataset


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        # Separate audio inputs from label inputs
        input_features = [
            {"input_features": self.processor.feature_extractor(
                f["audio"]["array"],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features[0]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [
            {"input_ids": self.processor.tokenizer(f["sentence"]).input_ids}
            for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # Replace pad token id with -100 so loss ignores padding
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove BOS token if prepended by tokenizer
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Metric ─────────────────────────────────────────────────────────────────────

def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def train(manifest_csv: str | Path = "data/train_manifest.csv"):
    """Fine-tune Whisper-small and save checkpoints to TRAIN_OUTPUT_DIR."""
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

    # Force Hindi decoding; disable multilingual head so model learns target lang
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    datasets = build_hf_dataset(manifest_csv)
    print(f"Train: {len(datasets['train'])}  |  Dev: {len(datasets['test'])}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(TRAIN_OUTPUT_DIR),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,       # effective batch = 16
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(TRAIN_OUTPUT_DIR / "best_model"))
    processor.save_pretrained(str(TRAIN_OUTPUT_DIR / "best_model"))
    print("Training complete. Model saved to:", TRAIN_OUTPUT_DIR / "best_model")


if __name__ == "__main__":
    train("data/train_manifest.csv")
