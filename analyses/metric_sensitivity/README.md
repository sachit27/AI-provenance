# Distribution distance sensitivity

This module compares Wasserstein-1, Wasserstein-2 and Gaussian-kernel MMD² using stored embeddings. It makes no model calls. Execution examples are in [the analysis workflow](../README.md).

W1 uses Euclidean transport cost with uniform participant and target masses. W2 is the square root of the minimum transport cost with squared Euclidean ground distance. MMD² uses the biased V-statistic, including self-pairs. The Gaussian bandwidth is 0.5, 1 or 2 times the median positive participant distance from a seeded sample of at most 1,000 participants.

PCA is fitted on participants alone, with unwhitened components at 20, 50 and 100 dimensions. That common basis is used for the official target and all random references. The primary transport analysis fits PCA jointly on participants and official sentences, so its absolute W2 values can differ.

The reported comparisons use the first 20 stored exact-length random references for OpenAI and MPNet in both countries. Scottish Nomic runs contain only official targets. Canadian runs also compare semantic regions against the official passage. Ranking agreement is calculated within a topic, embedding model and dimension, across one official target and 20 random references.

`distances.csv` contains the target distances. `results.json` records the geometry, bandwidth sample, target metadata, input hashes, regional values and rank correlations. Reference outputs for the 22 reported configurations are archived in `results/reference/metric_sensitivity/results/`.

The optional `--dataset` adapter reads `records.jsonl`, `summary.json` and `embeddings/<backend>.npz` with numeric `records` and `summary` arrays. It verifies text fingerprints and model identities where available. `--target` or `--target-manifest` can supply additional embedded targets. Those extensions were not used for the reported official-versus-random comparisons.

Distances are descriptive. Compare rankings within the same geometry. Values from different providers, topics or dimensions have different scales. The 11 existing tests check transport definitions, kernel calculations and stale or mixed embedding metadata.
