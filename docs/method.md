# Method settings

## Article preparation

Each sentiment input is the whitespace-normalised title, two newline characters, and the whitespace-normalised full body. Case and punctuation remain intact. A record without usable body text is retained in the prepared table but excluded from modelling. Collection quality tiers do not filter the modelling input. Exact and near-duplicate flags are recorded without further row deletion at this stage.

## Sentiment classifiers

| Model | Checkpoint | Frozen revision |
| --- | --- | --- |
| DistilRoBERTa | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | `ae0eab9ad336d7d548e0efe394b07c04bcaf6e91` |
| DeBERTa-v3 | `logicalqubit/deberta-v3-large-world-news-sentiment-classifier` | `128498f6c8e07e4fd3a7bb778850284ca55d344f` |

VADER uses the NLTK implementation and one score for the complete prepared article: Positive if compound ≥ 0.05, Negative if compound ≤ −0.05, otherwise Neutral.

The transformer models use 64-token overlap. Their saved tokenizer/model limits are 512 tokens for DistilRoBERTa and 256 for DeBERTa, including special tokens. Default batch sizes are 16 and 4 respectively. Probabilities are averaged across windows with weights equal to newly covered content tokens. Class names are read from each checkpoint's configuration rather than assuming a shared numeric class order. No project-specific training is performed.

The three outputs are joined by article ID. The code reports class distributions, pairwise exact agreement, Cohen's kappa, and three-model Fleiss' kappa. In the saved results, VADER and DeBERTa agree on 732 articles and differ on 1,109. Final grouping retains those agreements and applies supplied labels to the disagreements. The frozen final-label table is an input to grouping, not a label set generated from the classifiers alone and not an independent evaluation dataset.

## Topic models

Both sentiment groups use normalised embeddings from `sentence-transformers/all-MiniLM-L6-v2`. The encoder uses its default token limit, with no explicit long-article chunking. Thus its embedding can emphasise the opening of a long article; this differs from the transformer sentiment stage.

UMAP uses 15 neighbours, five components, minimum distance 0, cosine distance and seed 42. HDBSCAN uses Euclidean distance and excess-of-mass selection. CountVectorizer uses unigrams and bigrams. The original domain stop words are in `config/topic_model.json`.

| Setting | Positive | Negative |
| --- | ---: | ---: |
| Minimum cluster size | 12 | 25 |
| Minimum samples | 5 | 8 |
| Minimum document frequency | 2 | 3 |

Probabilistic topic output is disabled. Topic IDs belong to each fitted model; Positive topic 2 and Negative topic 2 are different clusters. Fresh fits can change topic assignments and IDs across environments. The saved sentence encoder was identified by model name, without a frozen revision. This release preserves that setting and does not invent a historical revision. For exact historical membership counts, use the included saved assignments.

Topic `-1` remains an outlier category. Neutral articles do not enter either topic model. No numbered topic is deleted or merged by the analysis scripts.

## Keyword and visual diagnostics

The keyword-cleaning configuration records added stop words and explicit word-form mappings. These affect derived keyword displays, not the saved article assignments. Keyword occurrence is a document frequency within a topic. Lift compares this with the term's document frequency in the complete corresponding sentiment group, including outliers.

Topic maps apply PCA to mean article embeddings separately for each sentiment group. Cross-group similarity uses cosine similarity of centroids in the original embedding space, not distances in the two PCA maps. PCA coordinates from separate groups do not share a coordinate system.

Keyword tables and charts describe the original positive and negative clusters. They do not provide independent topic accuracy estimates. The public code covers the central sentiment-to-topic workflow and its computational diagnostics.

Optional co-occurrence networks select the top 15 terms by the derived keyword relevance score, count documents containing each pair, and keep positive edges at or above the 70th percentile with a minimum of two documents. Counts combine the recorded word variants for each normalised term. Node and edge tables accompany each interactive HTML network. The charts are rebuilt from the computations; their layout and keyword order need not be pixel-identical to the manuscript figures.
