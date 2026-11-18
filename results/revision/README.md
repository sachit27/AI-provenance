# Revision reference results

These files record the analyses reported in the revised manuscript. All paths below are relative to this directory.

| Directory | Files and use |
|---|---|
| `scotland_original/` | Six-question coverage and cross-fitted benchmark summary; restoration of the 86 screened answers |
| `llm_summaries/results/` | Budget compliance, coverage metrics, paired comparisons, participant-rank correlations, bottom-decile overlap and regional concordance |
| `llm_summaries/outputs/` | All 36 final generated summaries with model, seed, budget, protocol and input-hash metadata |
| `stance_groups/` | Aggregate closed-answer group sizes and coverage comparisons |
| `idea_audit/` | Automated label shares, judge agreement, synthetic intervention results, sensitivity and stance associations |
| `provenance_diagnostics/results/` | Aggregate controlled-test statistics for each corpus, topic and embedding model |
| `metric_sensitivity/results/` | Distance values and provenance for four Canadian and 18 Scottish configurations |

Q21 generated summaries remain in the archive to document the attempted experiment. They are excluded from the main controlled comparison. Synthetic stance tests retain both directions in their aggregate output, but symmetric reverse-direction cases are duplicates and must not be counted as independent evidence.

The idea audit is exploratory. Its synthetic tests show differential recognition of opposing and other ideas. The label `MISSED` is an automated decision; it is not a human-coded absence judgment.

The restoration analysis adds all 86 Scottish answers removed by relevance screening. Its coverage and group comparisons use OpenAI and MPNet. The cross-fitted benchmarks retain the screened sample.

Source response pages, response identifiers, quotations, intermediate claims and model caches are excluded. Government report passages are provided under their source terms. File metadata paths have been made relative to the project root; numerical results are unchanged. Code/input hashes inside historical outputs describe the original analytical run. The public code has path adaptations for the released repository.

`SHA256SUMS` records the files in this reference archive. Regeneration writes to the working directories under `revision/`, leaving this archive available for comparison.
