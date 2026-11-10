#!/usr/bin/env python3
"""Fit independent article-level BERTopic models for Positive and Negative news."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def document_signature(ids, documents, model_name):
    """Identify ordered documents and encoder before reusing cached embeddings."""
    payload = json.dumps(
        {"model": model_name, "rows": list(zip(ids, documents))},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_group(group_name, frame, settings, config, output_dir, reuse_embeddings=False):
    import numpy as np
    import pandas as pd
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP

    name = group_name.lower()
    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    documents = frame["analysis_text"].astype(str).tolist()
    signature = document_signature(
        frame["dataset_id"].astype(str).tolist(), documents, config["embedding_model"]
    )
    embeddings_path = folder / f"{name}_article_embeddings.npy"
    manifest_path = folder / f"{name}_embedding_manifest.json"
    encoder = SentenceTransformer(config["embedding_model"])
    if reuse_embeddings and embeddings_path.exists():
        if not manifest_path.exists():
            raise ValueError(f"{group_name}: cached embeddings require a document manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("document_sha256") != signature:
            raise ValueError(f"{group_name}: cached embeddings belong to different ordered documents")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise ValueError(f"{group_name}: cached embedding dimensions do not match documents")
    else:
        # Match the original encoder call. Long articles use the encoder's default token limit.
        embeddings = encoder.encode(
            documents, batch_size=config["embedding_batch_size"],
            show_progress_bar=True, normalize_embeddings=True,
        )
        np.save(embeddings_path, embeddings)
        manifest_path.write_text(json.dumps({
            "document_sha256": signature,
            "embedding_model": config["embedding_model"],
            "article_rows": len(documents),
            "encoder_max_seq_length": encoder.max_seq_length,
        }, indent=2) + "\n", encoding="utf-8")

    model = BERTopic(
        embedding_model=encoder,
        umap_model=UMAP(random_state=config["random_seed"], **config["umap"]),
        hdbscan_model=HDBSCAN(
            min_cluster_size=settings["min_cluster_size"],
            min_samples=settings["min_samples"], **config["hdbscan"],
        ),
        vectorizer_model=CountVectorizer(
            stop_words=sorted(set(ENGLISH_STOP_WORDS) | set(config["domain_stop_words"])),
            min_df=settings["vectorizer_min_df"], ngram_range=tuple(config["ngram_range"]),
        ),
        calculate_probabilities=False, verbose=True,
    )
    labels, _ = model.fit_transform(documents, embeddings)
    assignments = frame.copy()
    assignments["bertopic_group"] = group_name
    assignments["topic_id"] = labels
    topic_info = model.get_topic_info().copy()
    topic_info["sentiment_group"] = group_name
    topic_info["top_keywords"] = topic_info["Topic"].map(
        lambda topic: "; ".join(word for word, _ in (model.get_topic(int(topic)) or [])[:10])
    )
    topic_info["article_share_pct"] = (100 * topic_info["Count"] / len(assignments)).round(2)

    rows = []
    for topic in sorted(set(labels)):
        if topic == -1:
            continue
        documents_for_topic = model.get_representative_docs(topic=int(topic)) or []
        if isinstance(documents_for_topic, dict):
            documents_for_topic = documents_for_topic.get(int(topic), [])
        for rank, text in enumerate(documents_for_topic[:5], start=1):
            matches = assignments.loc[
                assignments.topic_id.eq(topic) & assignments.analysis_text.eq(text)
            ]
            if matches.empty:
                continue
            row = matches.iloc[0]
            rows.append({
                "sentiment_group": group_name, "topic_id": int(topic),
                "representative_rank": rank, "dataset_id": row.dataset_id,
                "title": row.title, "body_text": row.body_text,
                "canonical_url": row.get("canonical_url", ""),
            })
    representatives = pd.DataFrame(rows, columns=[
        "sentiment_group", "topic_id", "representative_rank", "dataset_id",
        "title", "body_text", "canonical_url",
    ])
    assignments.to_csv(folder / f"{name}_article_topic_assignments.csv", index=False)
    topic_info.to_csv(folder / f"{name}_complete_topic_list.csv", index=False)
    representatives.to_csv(folder / f"{name}_representative_articles.csv", index=False)
    model.save(folder / "bertopic_model", serialization="safetensors", save_ctfidf=True)
    metrics = {
        "sentiment_group": group_name,
        "article_rows": len(assignments),
        "substantive_topics": int(assignments.loc[assignments.topic_id.ne(-1), "topic_id"].nunique()),
        "outlier_articles": int(assignments.topic_id.eq(-1).sum()),
        "outlier_percent": round(100 * assignments.topic_id.eq(-1).mean(), 2),
        **settings, "embedding_model": config["embedding_model"],
        "encoder_max_seq_length": encoder.max_seq_length,
        "random_seed": config["random_seed"], "topics_excluded": 0,
    }
    (folder / f"{name}_bertopic_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/final_articles.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/topics")
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic_model.json")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Reuse only caches with matching ordered-document signatures")
    args = parser.parse_args()
    import pandas as pd

    config = json.loads(args.config.read_text(encoding="utf-8"))
    articles = pd.read_csv(args.input, keep_default_na=False, dtype={"dataset_id": "string"})
    required = {"dataset_id", "title", "body_text", "analysis_text", "final_sentiment_label"}
    if missing := required - set(articles):
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if articles.dataset_id.eq("").any() or articles.dataset_id.duplicated().any():
        raise ValueError("dataset_id must be nonempty and unique")
    if articles.body_text.str.strip().eq("").any() or articles.analysis_text.str.strip().eq("").any():
        raise ValueError("All modelling articles require body_text and analysis_text")
    if not articles.final_sentiment_label.isin(["Positive", "Neutral", "Negative"]).all():
        raise ValueError("final_sentiment_label must be Positive, Neutral, or Negative")
    metrics = []
    for group, settings in config["groups"].items():
        frame = articles.loc[articles.final_sentiment_label.eq(group)].reset_index(drop=True)
        minimum = max(settings["min_cluster_size"], config["umap"]["n_components"] + 2)
        if len(frame) < minimum:
            raise ValueError(f"{group}: {len(frame)} articles; configured model needs at least {minimum}")
        metrics.append(fit_group(group, frame, settings, config, args.output_dir, args.reuse_embeddings))
    pd.DataFrame(metrics).to_csv(args.output_dir / "topic_comparison.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
