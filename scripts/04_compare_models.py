#!/usr/bin/env python3
"""Compare three classifiers on identical article IDs using distributions and agreement."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]
MODEL_COLUMNS = {"VADER": "vader_label", "DistilRoBERTa": "distilroberta_label", "DeBERTa-v3": "deberta_label"}


def cohen_kappa(first, second):
    observed = float(first.eq(second).mean())
    expected = sum(float(first.eq(label).mean()) * float(second.eq(label).mean()) for label in LABELS)
    return float((observed - expected) / (1 - expected)) if expected < 1 else None


def fleiss_kappa(label_frame):
    import numpy as np
    counts = np.column_stack([label_frame.eq(label).sum(axis=1).to_numpy() for label in LABELS])
    raters = label_frame.shape[1]
    observed = ((counts ** 2).sum(axis=1) - raters) / (raters * (raters - 1))
    expected = float(((counts.sum(axis=0) / counts.sum()) ** 2).sum())
    return float((observed.mean() - expected) / (1 - expected)) if expected < 1 else None


def compare_models(frames, articles=None):
    """Join by stable ID, rejecting duplicates, missing IDs, and unknown labels."""
    import pandas as pd
    if len(frames) != len(MODEL_COLUMNS):
        raise ValueError("Exactly three model tables are required")
    expected_ids = None
    comparison = None
    for (name, label_column), frame in zip(MODEL_COLUMNS.items(), frames):
        required = {"dataset_id", label_column}
        if required - set(frame.columns):
            raise ValueError(f"{name} missing fields: {sorted(required - set(frame.columns))}")
        if frame.empty or frame["dataset_id"].duplicated().any() or frame["dataset_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must contain nonempty unique dataset IDs")
        if not frame[label_column].isin(LABELS).all():
            raise ValueError(f"Unexpected labels in {label_column}")
        ids = set(frame["dataset_id"])
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{name} article IDs differ from the other models")
        fields = ["dataset_id", *[column for column in frame.columns if column.startswith(label_column.removesuffix("label"))]]
        part = frame[fields].copy()
        comparison = part if comparison is None else comparison.merge(part, on="dataset_id", validate="one_to_one")
    if articles is not None:
        if "eligible_for_model" in articles:
            articles = articles.loc[articles["eligible_for_model"].astype(str).str.casefold().eq("true")].copy()
        if articles["dataset_id"].duplicated().any() or set(articles["dataset_id"]) != expected_ids:
            raise ValueError("Prepared article IDs differ from model outputs")
        articles = articles.drop(columns=[column for column in comparison if column != "dataset_id" and column in articles])
        comparison = articles.merge(comparison, on="dataset_id", validate="one_to_one")
    columns = list(MODEL_COLUMNS.values())
    comparison["vader_distilroberta_agree"] = comparison[columns[0]].eq(comparison[columns[1]])
    comparison["vader_deberta_agree"] = comparison[columns[0]].eq(comparison[columns[2]])
    comparison["distilroberta_deberta_agree"] = comparison[columns[1]].eq(comparison[columns[2]])
    comparison["distinct_label_count"] = comparison[columns].nunique(axis=1)
    distribution = pd.DataFrame([
        {"model": name, "label": label, "articles": int(comparison[column].eq(label).sum()),
         "percent": 100 * float(comparison[column].eq(label).mean())}
        for name, column in MODEL_COLUMNS.items() for label in LABELS])
    pairwise = pd.DataFrame([
        {"model_a": first_name, "model_b": second_name,
         "agreement_count": int(comparison[first].eq(comparison[second]).sum()),
         "agreement_percent": 100 * float(comparison[first].eq(comparison[second]).mean()),
         "cohen_kappa": cohen_kappa(comparison[first], comparison[second])}
        for (first_name, first), (second_name, second) in itertools.combinations(MODEL_COLUMNS.items(), 2)])
    metrics = {"article_rows": len(comparison),
               "all_three_agree_count": int(comparison["distinct_label_count"].eq(1).sum()),
               "two_models_agree_count": int(comparison["distinct_label_count"].eq(2).sum()),
               "three_way_tie_count": int(comparison["distinct_label_count"].eq(3).sum()),
               "vader_deberta_agreement_count": int(comparison["vader_deberta_agree"].sum()),
               "vader_deberta_disagreement_count": int((~comparison["vader_deberta_agree"]).sum()),
               "fleiss_kappa_three_models": fleiss_kappa(comparison[columns]),
               "measure": "agreement between predictions; not accuracy"}
    return comparison, distribution, pairwise, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vader", type=Path, default=ROOT / "data/processed/vader.csv")
    parser.add_argument("--distilroberta", type=Path, default=ROOT / "data/processed/distilroberta.csv")
    parser.add_argument("--deberta", type=Path, default=ROOT / "data/processed/deberta.csv")
    parser.add_argument("--input", type=Path, help="Optional prepared corpus to include article fields")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/comparison")
    parser.add_argument("--expect-rows", type=int, help="Expected compared rows; use 1841 for the paper corpus")
    args = parser.parse_args()
    import pandas as pd
    paths = [args.vader, args.distilroberta, args.deberta]
    frames = [pd.read_csv(path, dtype={"dataset_id": "string"}, keep_default_na=False) for path in paths]
    articles = pd.read_csv(args.input, dtype={"dataset_id": "string"}, keep_default_na=False) if args.input else None
    comparison, distribution, pairwise, metrics = compare_models(frames, articles)
    if args.expect_rows is not None and len(comparison) != args.expect_rows:
        raise ValueError(f"Expected {args.expect_rows} rows, found {len(comparison)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    distribution.to_csv(args.output_dir / "label_distribution.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_agreement.csv", index=False)
    for first, second in itertools.combinations(MODEL_COLUMNS.values(), 2):
        matrix = pd.crosstab(comparison[first], comparison[second]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        matrix.to_csv(args.output_dir / f"{first.removesuffix('_label')}_vs_{second.removesuffix('_label')}.csv")
    metrics["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in zip(MODEL_COLUMNS, paths)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
