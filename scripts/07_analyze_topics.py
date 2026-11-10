#!/usr/bin/env python3
"""Create keyword tables, topic-size charts, centroid maps and similarity matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def topic_name(topics, topic_id):
    row = topics.loc[topics.Topic.eq(topic_id)]
    if row.empty:
        return ""
    if topic_id == -1:
        return "Outlier bucket"
    field = "provisional_name" if "provisional_name" in row else "Name"
    return str(row.iloc[0][field])


def raw_top_terms(topics):
    return {
        int(row.Topic): [term.strip().casefold() for term in str(row.top_keywords).split(";") if term.strip()]
        for row in topics[["Topic", "top_keywords"]].itertuples(index=False)
    }


def build_normalized_diagnostics(sentiment, assignments, topics, min_df, cleaning):
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    excluded_terms = set(cleaning["excluded_terms"])
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | set(cleaning["domain_stop_words"]) | excluded_terms)
    def canonicalize(term):
        return cleaning["word_forms"].get(term.casefold(), term.casefold())
    docs = assignments.analysis_text.astype(str).tolist()
    labels = assignments.topic_id.to_numpy()
    vectorizer = CountVectorizer(
        stop_words=stop_words, min_df=min_df, ngram_range=(1, 2), lowercase=True
    )
    raw_matrix = vectorizer.fit_transform(docs)
    raw_terms = np.asarray(vectorizer.get_feature_names_out())
    top_terms = raw_top_terms(topics)

    canonical_to_raw: dict[str, list[int]] = defaultdict(list)
    for index, term in enumerate(raw_terms):
        canonical_to_raw[canonicalize(term)].append(index)

    raw_corpus_occurrences = np.asarray(raw_matrix.sum(axis=0)).ravel().astype(int)
    raw_corpus_docs = np.asarray((raw_matrix > 0).sum(axis=0)).ravel().astype(int)

    records = []
    for topic_id in sorted(np.unique(labels)):
        mask = labels == topic_id
        n_topic = int(mask.sum())
        topic_matrix = raw_matrix[mask]
        raw_topic_occurrences = np.asarray(topic_matrix.sum(axis=0)).ravel().astype(int)
        raw_topic_docs = np.asarray((topic_matrix > 0).sum(axis=0)).ravel().astype(int)
        for canonical_term, indices in canonical_to_raw.items():
            topic_occurrences = int(raw_topic_occurrences[indices].sum())
            if not topic_occurrences:
                continue
            corpus_occurrences = int(raw_corpus_occurrences[indices].sum())
            # Avoid double-counting documents that contain both singular and plural forms.
            topic_docs = int(np.asarray((topic_matrix[:, indices].sum(axis=1) > 0)).sum())
            corpus_docs = int(np.asarray((raw_matrix[:, indices].sum(axis=1) > 0)).sum())
            topic_rate = 100 * topic_docs / n_topic
            corpus_rate = 100 * corpus_docs / len(docs)
            lift = topic_rate / corpus_rate if corpus_rate else np.nan
            outside_docs = corpus_docs - topic_docs
            outside_articles = len(docs) - n_topic
            outside_rate = 100 * outside_docs / outside_articles if outside_articles else 0.0
            outside_occurrences = corpus_occurrences - topic_occurrences
            topic_total_occurrences = int(raw_topic_occurrences.sum())
            corpus_total_occurrences = int(raw_corpus_occurrences.sum())
            log_odds = math.log(
                (topic_occurrences + 0.5)
                / (topic_total_occurrences - topic_occurrences + 0.5)
            ) - math.log(
                (outside_occurrences + 0.5)
                / (corpus_total_occurrences - topic_total_occurrences - outside_occurrences + 0.5)
            )
            raw_variant_counts = {
                raw_terms[j]: int(raw_topic_occurrences[j]) for j in indices
                if raw_topic_occurrences[j] > 0
            }
            raw_variant_docs = {
                raw_terms[j]: int(raw_topic_docs[j]) for j in indices
                if raw_topic_docs[j] > 0
            }
            raw_top = top_terms.get(topic_id, [])
            records.append({
                "Sentiment": sentiment,
                "Topic": topic_id,
                "Topic name": topic_name(topics, topic_id),
                "Topic label": f"{sentiment} Topic {topic_id} — {topic_name(topics, topic_id)}",
                "Canonical term": canonical_term,
                "Raw variants": "; ".join(raw_variant_counts),
                "Raw occurrence counts in Topic": json.dumps(raw_variant_counts, ensure_ascii=False),
                "Raw article counts in Topic": json.dumps(raw_variant_docs, ensure_ascii=False),
                "Occurrences in Topic": topic_occurrences,
                "Topic articles containing canonical term": topic_docs,
                "Topic article occurrence %": round(topic_rate, 2),
                "Occurrences in corpus": corpus_occurrences,
                "Corpus articles containing canonical term": corpus_docs,
                "Whole-corpus article occurrence %": round(corpus_rate, 2),
                "Outside-Topic article occurrence %": round(outside_rate, 2),
                "Topic vs corpus lift": round(lift, 3),
                "Exclusivity": round(topic_occurrences / corpus_occurrences, 3),
                "Log odds": round(log_odds, 3),
                "Current raw BERTopic top term": any(
                    canonicalize(term) == canonical_term and term not in excluded_terms
                    for term in raw_top
                ),
                "Normalization action": (
                    "merged explicit word forms" if len(raw_variant_counts) > 1 else "retained unchanged"
                ),
                "Diagnostic note": (
                    "high Topic specificity — retain" if topic_rate >= 10 and lift >= 1.5
                    else "corpus-wide / low specificity"
                    if corpus_rate >= 20 and lift <= 1.2
                    else "retain for interpretation"
                ),
            })

    result = pd.DataFrame(records)
    result["Relevance"] = (
        cleaning["relevance_lambda"] * np.log1p(result["Occurrences in Topic"])
        + (1 - cleaning["relevance_lambda"]) * np.log(result["Topic vs corpus lift"].clip(lower=0.001))
    )
    return result.sort_values(
        ["Topic", "Relevance"], ascending=[True, False]
    )


def load_group(topic_dir, sentiment):
    import numpy as np
    import pandas as pd

    name = sentiment.lower()
    folder = topic_dir / name
    assignments = pd.read_csv(folder / f"{name}_article_topic_assignments.csv", keep_default_na=False,
                              dtype={"dataset_id": "string"})
    topics = pd.read_csv(folder / f"{name}_complete_topic_list.csv", keep_default_na=False)
    embeddings = np.load(folder / f"{name}_article_embeddings.npy", allow_pickle=False)
    if assignments.dataset_id.duplicated().any():
        raise ValueError(f"{sentiment}: duplicate dataset_id values")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(assignments):
        raise ValueError(f"{sentiment}: embeddings must align with assignment rows")
    manifest_path = folder / f"{name}_embedding_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.dumps({
            "model": manifest["embedding_model"],
            "rows": list(zip(assignments.dataset_id.astype(str), assignments.analysis_text.astype(str))),
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != manifest.get("document_sha256"):
            raise ValueError(f"{sentiment}: assignment order or text differs from embedding manifest")
    assignments["topic_id"] = assignments.topic_id.astype(int)
    topics["Topic"] = topics.Topic.astype(int)
    observed = assignments.groupby("topic_id").size().to_dict()
    if observed != dict(zip(topics.Topic, topics.Count)):
        raise ValueError(f"{sentiment}: topic counts do not match article assignments")
    return assignments, topics, embeddings


def term_network(assignments, diagnostics, topic_id, min_df, cleaning, top_terms=15):
    """Count document-level co-occurrence after unioning each term's explicit variants."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    if not assignments.topic_id.eq(topic_id).any():
        raise ValueError(f"Topic {topic_id} is not present in this sentiment model")
    vectorizer = CountVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | set(cleaning["domain_stop_words"]) | set(cleaning["excluded_terms"])),
        min_df=min_df, ngram_range=(1, 2), lowercase=True,
    )
    matrix = vectorizer.fit_transform(assignments.analysis_text.astype(str))
    vocab = vectorizer.vocabulary_
    chosen = diagnostics.loc[diagnostics.Topic.eq(topic_id)].nlargest(top_terms, "Relevance").copy()
    chosen["variants"] = chosen["Raw variants"].map(
        lambda value: [term.strip() for term in str(value).split(";") if term.strip() in vocab]
    )
    chosen = chosen.loc[chosen.variants.map(bool)]
    names = chosen["Canonical term"].tolist()
    if not names:
        raise ValueError(f"Topic {topic_id}: no terms available for the network")
    mask = assignments.topic_id.eq(topic_id).to_numpy()
    presence = np.column_stack([
        np.asarray((matrix[mask][:, [vocab[term] for term in variants]].sum(axis=1) > 0)).ravel()
        for variants in chosen.variants
    ]).astype(int)
    co = presence.T @ presence
    positive_edges = co[np.triu_indices_from(co, k=1)]
    cutoff = max(2, int(np.percentile(positive_edges[positive_edges > 0], 70))) if np.any(positive_edges > 0) else 2
    nodes = pd.DataFrame({"term": names, "articles": np.diag(co)})
    edges = pd.DataFrame([
        {"left": names[i], "right": names[j], "articles": int(co[i, j])}
        for i in range(len(names)) for j in range(i + 1, len(names)) if co[i, j] >= cutoff
    ], columns=["left", "right", "articles"])
    return nodes, edges, pd.DataFrame(co, index=names, columns=names), cutoff


def draw_network(sentiment, topic_id, nodes, edges, output):
    import numpy as np
    import networkx as nx
    import plotly.graph_objects as go

    graph = nx.Graph()
    for row in nodes.itertuples():
        graph.add_node(row.term, size=int(row.articles))
    for row in edges.itertuples():
        graph.add_edge(row.left, row.right, weight=int(row.articles))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    positions = dict(nx.circular_layout(graph.subgraph(components[0]), scale=1.0))
    peripheral = [node for component in components[1:] for node in component]
    for x, node in zip(np.linspace(-.8, .8, max(len(peripheral), 1)), peripheral):
        positions[node] = np.array([x, -1.38])
    fig = go.Figure()
    maximum = max(nodes.articles.max(), 1)
    for left, right, data in graph.edges(data=True):
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="rgba(95,95,95,.45)", width=1 + 3 * data["weight"] / maximum),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[(x0+x1)/2], y=[(y0+y1)/2], mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0)"),
            hovertext=[f"{left} / {right}<br>Co-occurring articles: {data['weight']}"],
            hoverinfo="text", showlegend=False,
        ))
    names = list(graph.nodes)
    counts = np.array([graph.nodes[name]["size"] for name in names])
    label_positions = []
    for name in names:
        x, y = positions[name]
        label_positions.append("bottom center" if name in peripheral else
                               "middle right" if x > .55 else "middle left" if x < -.55 else
                               "top center" if y >= 0 else "bottom center")
    fig.add_trace(go.Scatter(
        x=[positions[name][0] for name in names], y=[positions[name][1] for name in names],
        mode="markers+text", text=names, textposition=label_positions,
        marker=dict(size=24 + 38 * np.sqrt(counts / max(counts.max(), 1)),
                    color="#86C5A8" if sentiment == "Positive" else "#E89B73", opacity=.88),
        hovertext=[f"{name}<br>Articles containing term: {graph.nodes[name]['size']}" for name in names],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(title=f"{sentiment} Topic {topic_id}: term co-occurrence",
                      template="plotly_white", height=900, dragmode="pan",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), hovermode="closest")
    fig.write_html(output, include_plotlyjs=True,
                   config={"scrollZoom": True, "displayModeBar": True, "responsive": True})


def centroid_tables(sentiment, assignments, embeddings, output_dir):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    ids = sorted(assignments.topic_id.unique())
    labels = [f"{sentiment} Topic {topic}" for topic in ids]
    centroids = np.vstack([
        embeddings[assignments.topic_id.eq(topic).to_numpy()].mean(axis=0) for topic in ids
    ])
    # The original map included Topic -1. Similarity matrices describe numbered topics.
    if min(centroids.shape) >= 2:
        xy = PCA(n_components=2, random_state=42).fit_transform(centroids)
    else:
        xy = np.zeros((len(centroids), 2))
    points = pd.DataFrame({
        "sentiment_group": sentiment, "topic_id": ids, "topic_label": labels,
        "articles": [int(assignments.topic_id.eq(topic).sum()) for topic in ids],
        "pca_1": xy[:, 0], "pca_2": xy[:, 1],
    })
    points.to_csv(output_dir / f"{sentiment.lower()}_topic_centroids_pca.csv", index=False)
    formal_mask = np.array([topic != -1 for topic in ids])
    formal_centroids = centroids[formal_mask]
    formal_labels = [label for label, formal in zip(labels, formal_mask) if formal]
    similarities = pd.DataFrame(
        cosine_similarity(formal_centroids) if len(formal_centroids) else [],
        index=formal_labels, columns=formal_labels,
    )
    similarities.to_csv(output_dir / f"{sentiment.lower()}_topic_similarity.csv", index_label="topic")
    return points, formal_centroids, formal_labels, similarities


def draw_group(sentiment, topics, diagnostics, points, similarities, output_dir, top_terms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = "#009E73" if sentiment == "Positive" else "#D55E00"
    name = sentiment.lower()
    data = topics.sort_values("Count")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([f"{sentiment} Topic {t}" for t in data.Topic], data.Count,
            color=["#999999" if t == -1 else color for t in data.Topic])
    ax.set(xlabel="Articles", title=f"{sentiment}: complete topic sizes")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_topic_sizes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(points.pca_1, points.pca_2, s=points.articles * 8, color=color, alpha=.65)
    for row in points.itertuples():
        ax.annotate(str(row.topic_id), (row.pca_1, row.pca_2), ha="center", va="center")
    ax.set(xlabel="Principal component 1", ylabel="Principal component 2",
           title=f"{sentiment}: topic centroid map (Topic -1 retained)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_topic_map.png", dpi=180)
    plt.close(fig)
    if not similarities.empty:
        draw_heatmap(similarities, f"{sentiment}: topic centroid similarity",
                     output_dir / f"{name}_topic_similarity.png")

    for row in topics.loc[topics.Topic.ne(-1)].itertuples(index=False):
        chosen = diagnostics.loc[diagnostics.Topic.eq(row.Topic)].nlargest(top_terms, "Relevance")
        chosen = chosen.sort_values("Relevance")
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(chosen["Canonical term"], chosen["Topic article occurrence %"], color=color)
        ax.bar_label(bars, labels=[
            f"{rate:.1f}% | lift {lift:.1f}"
            for rate, lift in zip(chosen["Topic article occurrence %"], chosen["Topic vs corpus lift"])
        ], padding=3, fontsize=8)
        ax.set(xlabel="Topic articles containing term (%)", xlim=(0, 125),
               title=f"{sentiment} Topic {row.Topic}: cleaned keyword profile (n={row.Count})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_topic_{row.Topic}_keywords.png", dpi=180)
        plt.close(fig)


def draw_heatmap(matrix, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(9, .8 * len(matrix.columns)), max(7, .55 * len(matrix))))
    im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if value > .55 else "white")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/topics")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/topic_analysis")
    parser.add_argument("--config", type=Path, default=ROOT / "config/keyword_cleaning.json")
    parser.add_argument("--topic-config", type=Path, default=ROOT / "config/topic_model.json")
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--network", action="append", default=[], metavar="SENTIMENT:TOPIC",
                        help="Export a chosen topic network, e.g. positive:4; repeat for more topics")
    args = parser.parse_args()
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    cleaning = json.loads(args.config.read_text(encoding="utf-8"))
    topic_config = json.loads(args.topic_config.read_text(encoding="utf-8"))
    requested_networks = []
    for value in args.network:
        try:
            sentiment, topic = value.split(":")
            sentiment = sentiment.capitalize()
            if sentiment not in topic_config["groups"]:
                raise ValueError
            requested_networks.append((sentiment, int(topic)))
        except ValueError:
            parser.error(f"Invalid --network {value!r}; use positive:4 or negative:3")
    if not 0 <= cleaning["relevance_lambda"] <= 1:
        raise ValueError("relevance_lambda must be in [0, 1]")
    if args.top_terms < 1:
        raise ValueError("top-terms must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centroids = {}
    summaries = []
    for sentiment, settings in topic_config["groups"].items():
        assignments, topics, embeddings = load_group(args.input_dir, sentiment)
        diagnostic = build_normalized_diagnostics(
            sentiment, assignments, topics, settings["vectorizer_min_df"], cleaning,
        )
        diagnostic.to_csv(args.output_dir / f"{sentiment.lower()}_term_diagnostics_normalized.csv", index=False)
        for group, topic_id in requested_networks:
            if group != sentiment:
                continue
            nodes, edges, co, cutoff = term_network(
                assignments, diagnostic, topic_id, settings["vectorizer_min_df"], cleaning, args.top_terms,
            )
            prefix = f"{sentiment.lower()}_topic_{topic_id}_network"
            nodes.to_csv(args.output_dir / f"{prefix}_nodes.csv", index=False)
            edges.to_csv(args.output_dir / f"{prefix}_edges.csv", index=False)
            co.to_csv(args.output_dir / f"{prefix}_cooccurrence.csv", index_label="term")
            (args.output_dir / f"{prefix}_settings.json").write_text(
                json.dumps({"sentiment": sentiment, "topic_id": topic_id,
                            "top_terms": args.top_terms, "minimum_edge_articles": cutoff}, indent=2) + "\n",
                encoding="utf-8",
            )
            if not args.no_figures:
                draw_network(sentiment, topic_id, nodes, edges, args.output_dir / f"{prefix}.html")
        points, centers, labels, similarity = centroid_tables(sentiment, assignments, embeddings, args.output_dir)
        centroids[sentiment] = (centers, labels)
        summaries.append(topics[["Topic", "Count", "Name"]].assign(sentiment_group=sentiment))
        if not args.no_figures:
            draw_group(sentiment, topics, diagnostic, points, similarity, args.output_dir, args.top_terms)
    pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "topic_sizes.csv", index=False)
    positive, positive_labels = centroids["Positive"]
    negative, negative_labels = centroids["Negative"]
    if len(positive) and len(negative):
        cross = pd.DataFrame(cosine_similarity(positive, negative), index=positive_labels, columns=negative_labels)
        cross.to_csv(args.output_dir / "cross_sentiment_topic_similarity.csv", index_label="topic")
        if not args.no_figures:
            draw_heatmap(cross, "Positive / Negative topic centroid similarity",
                         args.output_dir / "cross_sentiment_topic_similarity.png")
    (args.output_dir / "keyword_cleaning_used.json").write_text(
        json.dumps(cleaning, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Topic tables and figures: {args.output_dir}")


if __name__ == "__main__":
    main()
