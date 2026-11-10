#!/usr/bin/env python3
"""Retain VADER–DeBERTa agreements and apply supplied labels to disagreements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Negative", "Neutral", "Positive")
ARTICLE_FIELDS = (
    "dataset_id", "source_system", "source_domain", "canonical_url", "title",
    "published_at", "summary", "body_text", "quality_tier", "analysis_text",
    "body_word_count", "analysis_word_count", "body_present", "eligible_for_model",
    "near_duplicate_cluster_id", "near_duplicate_flag",
)


def read_unique(path, required):
    import pandas as pd
    frame = pd.read_csv(path, dtype={"dataset_id": str}, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    if frame["dataset_id"].eq("").any() or frame["dataset_id"].duplicated().any():
        raise ValueError(f"{path}: dataset_id must be nonempty and unique")
    return frame


def build_groups(articles, vader, deberta, decisions):
    """Require matching IDs and a supplied decision for every disagreement.

    A decision table may contain only disagreements or the complete frozen label
    set. A complete table must also match every retained model agreement.
    """
    import pandas as pd
    required = {"dataset_id", "body_text", "analysis_text"}
    if required - set(articles.columns):
        raise ValueError(f"Article input missing {sorted(required - set(articles.columns))}")
    frames = [(articles, None), (vader, "vader_label"),
              (deberta, "deberta_label"), (decisions, "final_sentiment_label")]
    for frame, label in frames:
        if "dataset_id" not in frame or frame["dataset_id"].eq("").any() or frame["dataset_id"].duplicated().any():
            raise ValueError("Every input must have nonempty, unique dataset_id values")
        if label and (label not in frame or not frame[label].isin(LABELS).all()):
            raise ValueError(f"Invalid or missing labels in {label}")
    eligible = articles.loc[articles["body_text"].astype(str).str.strip().ne("")].copy()
    if eligible.empty:
        raise ValueError("No articles have usable body text")
    ids = set(eligible["dataset_id"])
    for name, frame in [("VADER", vader), ("DeBERTa", deberta)]:
        if set(frame["dataset_id"]) != ids:
            raise ValueError(f"{name} IDs do not exactly match body-present article IDs")
    if set(decisions["dataset_id"]) - ids:
        raise ValueError("Decision table contains IDs outside the modelling corpus")
    columns = [column for column in ARTICLE_FIELDS if column in eligible]
    result = eligible[columns].merge(
        vader[["dataset_id", "vader_label"]], on="dataset_id", validate="one_to_one"
    ).merge(deberta[["dataset_id", "deberta_label"]], on="dataset_id", validate="one_to_one")
    agree = result["vader_label"].eq(result["deberta_label"])
    decision_map = decisions.set_index("dataset_id")["final_sentiment_label"]
    supplied = result["dataset_id"].map(decision_map)
    if (agree & supplied.notna() & supplied.ne(result["vader_label"])).any():
        raise ValueError("A supplied label contradicts a retained VADER–DeBERTa agreement")
    unresolved = (~agree) & supplied.isna()
    if unresolved.any():
        ids_missing = result.loc[unresolved, "dataset_id"].tolist()
        raise ValueError(f"Missing decisions for {len(ids_missing)} disagreements: {ids_missing[:5]}")
    result["final_sentiment_label"] = result["vader_label"].where(agree, supplied)
    summary = {
        "input_records": int(len(articles)), "modelled_articles": int(len(result)),
        "excluded_missing_body": int(len(articles) - len(result)),
        "vader_deberta_agreements": int(agree.sum()),
        "vader_deberta_disagreements": int((~agree).sum()),
        "final_counts": {label: int(result["final_sentiment_label"].eq(label).sum()) for label in LABELS},
    }
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/article_input.csv")
    parser.add_argument("--vader", type=Path, default=ROOT / "data/processed/vader.csv")
    parser.add_argument("--deberta", type=Path, default=ROOT / "data/processed/deberta.csv")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/reference/final_labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/final_articles.csv")
    args = parser.parse_args()
    articles = read_unique(args.input, {"dataset_id", "body_text", "analysis_text"})
    vader = read_unique(args.vader, {"dataset_id", "vader_label"})
    deberta = read_unique(args.deberta, {"dataset_id", "deberta_label"})
    decisions = read_unique(args.decisions, {"dataset_id", "final_sentiment_label"})
    result, summary = build_groups(articles, vader, deberta, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    for label in LABELS:
        result.loc[result["final_sentiment_label"].eq(label)].to_csv(
            args.output.parent / f"{label.lower()}_articles.csv", index=False
        )
    (args.output.parent / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
