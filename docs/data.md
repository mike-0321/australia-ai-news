# Data and file interfaces

## Input articles

Place your input at `data/input/articles.csv` or pass `--input` to preparation. Each row requires a unique, nonempty `dataset_id`, a `title`, and a `body_text` column. Empty body values are allowed in the source corpus and are excluded from modelling.

Recommended metadata columns are `source_system`, `source_domain`, `canonical_url`, `published_at`, `summary`, and `quality_tier`. Nonempty publication dates must parse as timestamps. Read IDs as strings. Preparation creates `analysis_text`, body-presence flags, word counts and duplicate indicators without using `quality_tier` to remove rows.

The included `data/example/articles.csv` contains eight invented records with `example.invalid` URLs. Six have body text; two do not. They are software examples, not research evidence.

## Saved reference tables

`data/reference/` contains an explicit selection of non-text fields from saved study outputs:

| File | Contents |
| --- | --- |
| `vader.csv` | 1,841 IDs, labels and VADER scores |
| `distilroberta.csv` | 1,841 IDs, labels, class probabilities and token/chunk counts |
| `deberta.csv` | 1,841 IDs, labels, class probabilities and token/chunk counts |
| `final_labels.csv` | 1,841 IDs and frozen final sentiment labels |
| `positive_topic_assignments.csv` | 358 IDs and original topic IDs, including −1 |
| `negative_topic_assignments.csv` | 1,013 IDs and original topic IDs, including −1 |

`results/reference/` contains original topic counts and top keywords plus the main corpus counts. These tables contain neither article bodies nor article excerpts. They support recomputing saved counts and classifier agreement; rerunning text-based inference or keyword extraction requires the original article inputs.

## Final decisions

For a new corpus, provide a CSV with exactly the interface `dataset_id,final_sentiment_label`. Allowed labels are `Negative`, `Neutral`, and `Positive`. It can contain all eligible articles or just VADER–DeBERTa disagreements. Every disagreement must be covered. Duplicate IDs, unknown IDs, invalid labels, and a supplied label that contradicts a retained agreement cause an error.

## Generated files

| Command | Default input | Default output |
| --- | --- | --- |
| `00_collect_news.py` | Live APIs; optional GDELT candidate CSVs | `data/input/articles.csv` |
| `01_prepare_corpus.py` | `data/input/articles.csv` | `data/processed/article_input.csv` |
| `02_run_vader.py` | Prepared articles | `data/processed/vader.csv` |
| `03_run_transformers.py` | Prepared articles | `data/processed/distilroberta.csv`, `deberta.csv` |
| `04_compare_models.py` | Three model outputs | `results/comparison/` |
| `05_build_final_groups.py` | Prepared articles, VADER, DeBERTa, decisions | `data/processed/final_articles.csv` and three sentiment groups |
| `06_fit_topics.py` | Final articles | `results/topics/positive/`, `negative/` |
| `07_analyze_topics.py` | Saved topic outputs and ordered embeddings | `results/topic_analysis/` |
| `reproduce_reference.py` | Included reference labels and assignments | `results/reproduced/` |

Topic outputs include full article assignments, embeddings, topic information, representative articles and model files. Keep them local when the underlying article text cannot be redistributed. The provided Git ignore rules exclude these generated directories.

## Optional historical GDELT candidates

`00_collect_news.py --gdelt-input path/to/candidates.csv` passes an extra candidate file to the merge stage. Repeat the option for multiple files. It accepts the project's collection schema with `canonical_url`, `title`, and source fields, or the historical GDELT schema with `url`, `title`, `date`, `source_domain`, `country`, `source_guess`, `gdelt_query`, and `matched_terms`. In the latter format missing body text remains empty. The strict and expanded collectors run before merge unless explicitly skipped.

The saved corpus is a fixed historical input. Retrieving the same dates from a live index does not ensure the same candidate pool, order, or availability of article bodies.
