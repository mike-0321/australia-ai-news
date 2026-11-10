#!/usr/bin/env python3
"""Run frozen news-sentiment classifiers with full-token overlapping windows."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]
MODELS = {
    "distilroberta": {
        "id": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "revision": "ae0eab9ad336d7d548e0efe394b07c04bcaf6e91",
        "batch_size": 16,
    },
    "deberta": {
        "id": "logicalqubit/deberta-v3-large-world-news-sentiment-classifier",
        "revision": "128498f6c8e07e4fd3a7bb778850284ca55d344f",
        "batch_size": 4,
    },
}


def article_chunks(token_ids: list[int], content_window: int, stride: int = 64):
    """Return overlapping chunks weighted by newly covered content tokens."""
    if content_window <= 0 or not 0 <= stride < content_window:
        raise ValueError("stride must satisfy 0 <= stride < content_window")
    chunks, start = [], 0
    while start < len(token_ids):
        chunk = token_ids[start:start + content_window]
        newly_covered = len(chunk) if start == 0 else len(chunk) - stride
        chunks.append((chunk, newly_covered))
        if start + len(chunk) >= len(token_ids):
            break
        start += content_window - stride
    return chunks


def label_mapping(model):
    mapping = {int(index): str(label).strip().casefold().capitalize()
               for index, label in model.config.id2label.items()}
    if set(mapping.values()) != set(LABELS) or set(mapping) != set(range(3)):
        raise ValueError(f"Expected exactly three sentiment labels; found {mapping}")
    return mapping


def choose_device(requested: str):
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eligible(path: Path):
    import pandas as pd
    spec = importlib.util.spec_from_file_location("vader_stage", Path(__file__).with_name("02_run_vader.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df = pd.read_csv(path, dtype={"dataset_id": "string"}, keep_default_na=False)
    return df, module.eligible_articles(df)


def run_model(eligible, model_name: str, output_path: Path, input_path: Path,
              device_name: str = "auto", batch_size: int | None = None, stride: int = 64):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    config = MODELS[model_name]
    batch_size = config["batch_size"] if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = choose_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=config["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["id"], revision=config["revision"], use_safetensors=True)
    model.to(device).eval()
    mapping = label_mapping(model)
    tokenizer_limit = int(tokenizer.model_max_length)
    if model_name == "distilroberta":
        model_max_length = tokenizer_limit
    else:
        limits = [value for value in (tokenizer_limit, int(getattr(model.config, "max_position_embeddings", 512)))
                  if 0 < value <= 4096]
        if not limits:
            raise ValueError("No reasonable tokenizer/model maximum length")
        model_max_length = min(limits)
    content_window = model_max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
    if content_window <= 0 or model_max_length > 4096:
        raise ValueError(f"Unexpected model maximum length: {model_max_length}")
    token_counts, chunk_counts, records = [], [], []
    for index, value in enumerate(eligible["analysis_text"].astype(str)):
        tokens = tokenizer(value, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        chunks = article_chunks(tokens, content_window, stride)
        if not chunks:
            raise ValueError(f"Empty token sequence for {eligible.at[index, 'dataset_id']}")
        token_counts.append(len(tokens))
        chunk_counts.append(len(chunks))
        records.extend((index, ids, weight) for ids, weight in chunks)
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    end_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    if start_token is None or end_token is None:
        raise ValueError("Tokenizer must define BOS/CLS and EOS/SEP token IDs")
    probability_sums = np.zeros((len(eligible), len(LABELS)), dtype=np.float64)
    weight_sums = np.zeros(len(eligible), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            features = [{"input_ids": [start_token, *ids, end_token]} for _, ids, _ in batch_records]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**batch).logits, dim=-1).cpu().numpy()
            for probability, (index, _, weight) in zip(probabilities, batch_records):
                for model_column, value in enumerate(probability):
                    probability_sums[index, LABELS.index(mapping[model_column])] += float(value) * weight
                weight_sums[index] += weight
            if start == 0 or (start // batch_size + 1) % 100 == 0:
                print(f"{model_name}: chunks {min(start + batch_size, len(records)):,}/{len(records):,}", flush=True)
    if not np.array_equal(weight_sums, np.asarray(token_counts)):
        raise ValueError("Chunk weights do not cover each article exactly once")
    probabilities = probability_sums / weight_sums[:, None]
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Article probability rows are invalid")
    output = eligible[["dataset_id"]].copy()
    for column, label in enumerate(LABELS):
        output[f"{model_name}_{label.casefold()}_prob"] = probabilities[:, column]
    output[f"{model_name}_label"] = [LABELS[index] for index in probabilities.argmax(axis=1)]
    output[f"{model_name}_confidence"] = probabilities.max(axis=1)
    output[f"{model_name}_token_count"] = token_counts
    output[f"{model_name}_chunk_count"] = chunk_counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metrics = {
        "model_id": config["id"], "model_revision": config["revision"],
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "analysed_rows": len(output), "input_text": "title + full body_text",
        "model_max_length": model_max_length, "content_tokens_per_chunk": content_window,
        "overlap_tokens": stride, "batch_size": batch_size,
        "aggregation": "new-token-coverage-weighted mean of chunk probabilities",
        "total_article_tokens": sum(token_counts), "total_chunks": sum(chunk_counts),
        "label_mapping": mapping,
        "label_distribution": {label: int(output[f"{model_name}_label"].eq(label).sum()) for label in LABELS},
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "device": str(device),
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/article_input.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, help="Default: 16 for DistilRoBERTa, 4 for DeBERTa")
    parser.add_argument("--stride", type=int, default=64, help="Overlap in content tokens")
    parser.add_argument("--limit", type=int, help="Process the first N eligible rows for a smoke run")
    parser.add_argument("--expect-rows", type=int, help="Expected full eligible corpus size before --limit")
    args = parser.parse_args()
    _, eligible = load_eligible(args.input)
    if args.expect_rows is not None and len(eligible) != args.expect_rows:
        raise ValueError(f"Expected {args.expect_rows} eligible rows, found {len(eligible)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        eligible = eligible.head(args.limit).copy()
    for name in MODELS if args.model == "all" else [args.model]:
        run_model(eligible, name, args.output_dir / f"{name}.csv", args.input,
                  args.device, args.batch_size, args.stride)


if __name__ == "__main__":
    main()
