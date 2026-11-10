#!/usr/bin/env python3
"""Recompute paper summary tables and charts from the saved text-free labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]


def summarize(reference_dir):
    import pandas as pd
    final = pd.read_csv(reference_dir / "final_labels.csv", dtype={"dataset_id": str})
    if final["dataset_id"].isna().any() or final["dataset_id"].duplicated().any():
        raise ValueError("Final labels need unique, nonempty dataset_id values")
    ids = set(final["dataset_id"])
    rows = []
    for name in ["vader", "distilroberta", "deberta", "final"]:
        frame = final if name == "final" else pd.read_csv(reference_dir / f"{name}.csv", dtype={"dataset_id": str})
        field = "final_sentiment_label" if name == "final" else f"{name}_label"
        if frame["dataset_id"].duplicated().any() or set(frame["dataset_id"]) != ids:
            raise ValueError(f"{name}: mismatched or duplicate dataset IDs")
        if not frame[field].isin(LABELS).all():
            raise ValueError(f"{name}: invalid sentiment labels")
        for label in LABELS:
            count = int(frame[field].eq(label).sum())
            rows.append({"model": name, "label": label, "count": count,
                         "percent": 100 * count / len(frame)})
    coverage, sizes = [], []
    for group in ["positive", "negative"]:
        assignments = pd.read_csv(reference_dir / f"{group}_topic_assignments.csv", dtype={"dataset_id": str})
        group_ids = set(final.loc[final["final_sentiment_label"].eq(group.title()), "dataset_id"])
        if assignments["dataset_id"].duplicated().any() or set(assignments["dataset_id"]) != group_ids:
            raise ValueError(f"{group}: assignments do not match final sentiment membership")
        topic_ids = pd.to_numeric(assignments["topic_id"], errors="raise")
        if topic_ids.lt(-1).any() or topic_ids.mod(1).ne(0).any():
            raise ValueError(f"{group}: invalid topic IDs")
        counts = topic_ids.value_counts().sort_index()
        outliers = int(counts.get(-1, 0))
        coverage.append({"group": group, "articles": len(assignments),
                         "numbered_topics": int((counts.index >= 0).sum()),
                         "assigned": len(assignments) - outliers, "outliers": outliers})
        sizes.extend({"group": group, "topic_id": int(topic), "article_count": int(count)}
                     for topic, count in counts.items())
    return pd.DataFrame(rows), pd.DataFrame(coverage), pd.DataFrame(sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    distributions, coverage, sizes = summarize(args.reference_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distributions.to_csv(args.output_dir / "sentiment_distribution.csv", index=False)
    coverage.to_csv(args.output_dir / "topic_coverage.csv", index=False)
    sizes.to_csv(args.output_dir / "topic_sizes.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    table = distributions.pivot(index="model", columns="label", values="count").reindex(
        ["vader", "distilroberta", "deberta", "final"]
    )[LABELS]
    ax = table.plot.bar(stacked=True, color=["#c95b62", "#a2a9b3", "#368c76"], figsize=(8, 4.8), rot=0)
    ax.set(xlabel="", ylabel="Articles", title="Sentiment labels for the same article corpus")
    ax.legend(title="Sentiment", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(args.output_dir / "sentiment_distribution.png", bbox_inches="tight")
    plt.close()
    for group in ["positive", "negative"]:
        frame = sizes.loc[sizes["group"].eq(group)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(frame["topic_id"].astype(str), frame["article_count"],
               color=["#a2a9b3" if n == -1 else ("#368c76" if group == "positive" else "#c95b62") for n in frame["topic_id"]])
        ax.set(xlabel="Topic ID (−1 = outlier)", ylabel="Articles", title=f"{group.title()} topic sizes")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{group}_topic_sizes.png")
        plt.close(fig)
    print(json.dumps({"sentiment": distributions.to_dict("records"), "topics": coverage.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
