#!/usr/bin/env python3
"""Score complete title + body inputs with VADER's standard compound thresholds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def eligible_articles(df):
    required = {"dataset_id", "body_text", "analysis_text", "eligible_for_model"}
    if required - set(df.columns):
        raise ValueError(f"Missing prepared fields: {sorted(required - set(df.columns))}")
    if df["dataset_id"].astype(str).str.strip().eq("").any() or df["dataset_id"].duplicated().any():
        raise ValueError("dataset_id must be nonempty and unique")
    flags = df["eligible_for_model"].astype(str).str.casefold()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("eligible_for_model must contain only True/False")
    present = df["body_text"].astype(str).str.strip().ne("")
    if not flags.eq("true").eq(present).all():
        raise ValueError("eligible_for_model must match body-text availability")
    eligible = df.loc[flags.eq("true")].copy().reset_index(drop=True)
    if eligible.empty or eligible["analysis_text"].astype(str).str.strip().eq("").any():
        raise ValueError("At least one nonempty eligible analysis_text is required")
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/article_input.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/vader.csv")
    parser.add_argument("--expect-rows", type=int, help="Expected eligible rows; use 1841 for the paper corpus")
    args = parser.parse_args()
    import nltk
    import pandas as pd
    from nltk.sentiment import SentimentIntensityAnalyzer
    df = pd.read_csv(args.input, dtype={"dataset_id": "string"}, keep_default_na=False)
    eligible = eligible_articles(df)
    if args.expect_rows is not None and len(eligible) != args.expect_rows:
        raise ValueError(f"Expected {args.expect_rows} eligible rows, found {len(eligible)}")
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True, raise_on_error=True)
        analyzer = SentimentIntensityAnalyzer()
    scores = eligible["analysis_text"].map(analyzer.polarity_scores)
    output = eligible[["dataset_id"]].copy()
    for key in ("neg", "neu", "pos", "compound"):
        output[f"vader_{key}"] = scores.map(lambda score, k=key: float(score[k]))
    output["vader_label"] = output["vader_compound"].map(vader_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metrics = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "input_rows": len(df), "analysed_rows": len(output),
               "excluded_missing_body_rows": len(df) - len(output), "nltk_version": nltk.__version__,
               "input_text": "title + full body_text", "thresholds": {"positive": 0.05, "negative": -0.05},
               "label_distribution": {label: int(output["vader_label"].eq(label).sum()) for label in LABELS}}
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
