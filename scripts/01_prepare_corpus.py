#!/usr/bin/env python3
"""Prepare title + full-body inputs, retaining every source row and duplicate flag."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def normalise_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_key(value: object) -> str:
    value = re.sub(r"https?://(www\.)?", "", normalise_text(value).casefold())
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def word_count(value: object) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", normalise_text(value)))


def url_slug(value: object) -> str:
    path = urlsplit(normalise_text(value)).path.rstrip("/")
    return re.sub(r"\.html?$", "", path.rsplit("/", 1)[-1].casefold())


def publisher_story_id(value: object) -> str:
    match = re.search(r"/(?:story|news)/(\d{7,})(?:/|$)", urlsplit(normalise_text(value)).path.casefold())
    return match.group(1) if match else ""


def grouped_pairs(values):
    groups = defaultdict(list)
    for index, value in values.items():
        if value:
            groups[str(value)].append(int(index))
    return [(a, b) for indexes in groups.values() for i, a in enumerate(indexes) for b in indexes[i + 1:]]


def near_duplicate_candidates(df):
    """Flag character-ngram title neighbours and shared cross-domain story keys."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    columns = ["near_duplicate_cluster_id", "dataset_id_a", "dataset_id_b",
               "title_cosine_similarity", "detection_methods"]
    if len(df) < 2:
        return pd.DataFrame(columns=columns), {}
    titles = df["title"].map(normalise_text).tolist()
    methods, similarities = defaultdict(set), {}
    matrix = None
    try:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(titles)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
    if matrix is not None:
        model = NearestNeighbors(n_neighbors=min(5, len(df)), metric="cosine").fit(matrix)
        distances, neighbours = model.kneighbors(matrix)
        for i, (row_distances, row_neighbours) in enumerate(zip(distances, neighbours)):
            for distance, j in zip(row_distances, row_neighbours):
                if i == int(j):
                    continue
                a, b = sorted((i, int(j)))
                score = 1.0 - float(distance)
                if score >= 0.92:
                    methods[(a, b)].add("title_char_ngram_cosine>=0.92")
                    similarities[(a, b)] = max(score, similarities.get((a, b), 0.0))
    for key, method in ((url_slug, "shared_cross_domain_url_slug"),
                        (publisher_story_id, "shared_cross_domain_story_id")):
        for a, b in grouped_pairs(df["canonical_url"].map(key)):
            if df.at[a, "source_domain"] != df.at[b, "source_domain"]:
                methods[(a, b)].add(method)
    parent = list(range(len(df)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in methods:
        parent[find(b)] = find(a)
        if (a, b) not in similarities:
            similarities[(a, b)] = float(matrix[a].multiply(matrix[b]).sum()) if matrix is not None else 0.0
    groups = defaultdict(set)
    for a, b in methods:
        groups[find(a)].update((a, b))
    clusters = {}
    for number, indexes in enumerate(sorted(groups.values(), key=min), 1):
        for index in indexes:
            clusters[index] = f"ND{number:03d}"
    rows = [{"near_duplicate_cluster_id": clusters[a], "dataset_id_a": df.at[a, "dataset_id"],
             "dataset_id_b": df.at[b, "dataset_id"], "title_cosine_similarity": round(similarities[(a, b)], 4),
             "detection_methods": ";".join(sorted(methods[(a, b)]))} for a, b in sorted(methods)]
    return pd.DataFrame(rows, columns=columns), clusters


def prepare_corpus(df):
    """Return prepared articles and candidate pairs without removing source rows."""
    import pandas as pd

    required = {"dataset_id", "title", "body_text"}
    if required - set(df.columns):
        raise ValueError(f"Missing required fields: {sorted(required - set(df.columns))}")
    df = df.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("The input corpus is empty")
    if df["dataset_id"].astype(str).str.strip().eq("").any() or df["dataset_id"].duplicated().any():
        raise ValueError("dataset_id must be nonempty and unique")
    for column in ("summary", "canonical_url", "source_domain", "source_system"):
        if column not in df:
            df[column] = ""
    body, title = df["body_text"].map(normalise_text), df["title"].map(normalise_text)
    if "published_at" in df:
        dates = pd.to_datetime(df["published_at"], format="mixed", utc=True, errors="coerce")
        present = df["published_at"].astype(str).str.strip().ne("")
        if (present & dates.isna()).any():
            raise ValueError("Nonempty published_at values must be valid dates")
        df["published_at_utc"] = dates.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    df["body_present"] = body.ne("")
    df["body_word_count"] = body.map(word_count)
    df["summary_word_count"] = df["summary"].map(word_count)
    df["analysis_input_scope"] = df["body_present"].map({True: "title_and_body", False: "title_only_missing_body"})
    df["analysis_text"] = [f"{headline}\n\n{article_body}" if article_body else headline
                           for headline, article_body in zip(title, body)]
    df["analysis_word_count"] = df["analysis_text"].map(word_count)
    df["eligible_for_model"] = df["body_present"]
    df["quality_tier_used_for_filtering"] = False
    for name, key in (("dataset_id", df["dataset_id"].map(normalise_key)),
                      ("url", df["canonical_url"].map(normalise_key))):
        df[f"exact_duplicate_{name}"] = key.ne("") & key.duplicated(keep=False)
    key = title.map(normalise_key) + "||" + body.map(normalise_key)
    df["exact_duplicate_title_body"] = body.ne("") & key.duplicated(keep=False)
    df["exact_duplicate_flag"] = df[["exact_duplicate_dataset_id", "exact_duplicate_url", "exact_duplicate_title_body"]].any(axis=1)
    pairs, clusters = near_duplicate_candidates(df)
    df["near_duplicate_cluster_id"] = [clusters.get(i, "") for i in df.index]
    df["near_duplicate_flag"] = df["near_duplicate_cluster_id"].ne("")
    return df, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/input/articles.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/article_input.csv")
    parser.add_argument("--expect-rows", type=int, help="Optional check; use 2000 for the paper corpus")
    args = parser.parse_args()
    import pandas as pd
    source = pd.read_csv(args.input, dtype={"dataset_id": "string"}, keep_default_na=False)
    if args.expect_rows is not None and len(source) != args.expect_rows:
        raise ValueError(f"Expected {args.expect_rows} rows, found {len(source)}")
    articles, pairs = prepare_corpus(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    articles.to_csv(args.output, index=False)
    pairs.to_csv(args.output.with_name("near_duplicate_candidates.csv"), index=False)
    metrics = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "input_rows": len(articles), "eligible_rows": int(articles["eligible_for_model"].sum()),
               "missing_body_rows": int((~articles["body_present"]).sum()),
               "exact_duplicate_rows": int(articles["exact_duplicate_flag"].sum()),
               "near_duplicate_pairs": len(pairs), "rows_removed": 0,
               "quality_tier_used_for_filtering": False}
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
