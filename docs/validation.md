# Local validation

Checked on 2026-11-11 with Python 3.14.5. Top-level installed versions are recorded in `requirements-tested.txt`; this is an observed environment snapshot, not a complete transitive lockfile or a claim about the original experimental environment.

| Check | Result |
| --- | --- |
| Automated tests | 18 passed, including article identity, missing decisions, chunk coverage, keyword counting and embedding order |
| Preparation on the original 2,000 rows | Article text, eligibility and duplicate flags match saved outputs |
| Full VADER run on 1,841 articles | All labels and all four scores match saved outputs exactly |
| Transformer inference | Both pinned checkpoints loaded from local cache and passed a one-article CPU inference check |
| Final label assembly | 732 retained agreements; 1,109 supplied decisions; final counts 358 Positive, 470 Neutral, 1,013 Negative |
| Saved model comparison | Three-way agreement counts and all reported kappa values reproduced |
| BERTopic refit from original ordered embeddings | All 358 Positive and 1,013 Negative topic IDs, topic counts and top keyword strings match saved results |
| Keyword diagnostics | All 63,939 Positive and 101,782 Negative normalised term rows match saved occurrence, lift, exclusivity, log-odds and relevance values |
| Visual outputs | 26 topic/keyword/similarity PNGs and two interactive keyword networks generated successfully |
| Collection entry point | CLI help, dry run, argument forwarding and column export checked without live API calls |

The complete transformer corpus was not inferred again, and the topic encoder was not rerun over the full corpus. BERTopic validation reused the original article embeddings and refitted UMAP, HDBSCAN and topic representations. This verifies the extracted topic pipeline while preserving the original embeddings.

The original stop-word configuration includes the phrase `artificial intelligence`. CountVectorizer emits a stop-word/tokenisation warning for this phrase; it was retained to preserve the original settings. The checks still pass, and changing that setting could change the topic representations.

The public reference tables support the lightweight reproduction commands in the README. The full-data checks above used local article text and embeddings, which are not included in this repository.
