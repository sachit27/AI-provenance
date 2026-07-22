# Reference results

This directory contains compact, human-readable reference outputs for checking
a fresh run. These files are not inputs to the analysis pipeline.

## Files

- `crossfit_confirmatory_summary.json` contains the unrounded confirmatory
  benchmark estimates, conditional participant-level bootstrap intervals,
  paired-randomization results, repeated-partition summaries, chance
  comparisons, and worst-region diagnostics reported in the manuscript.
- `input_hashes.sha256` records the SHA-256 checksums of the two public source
  CSV files in `data/`.

## Terms used in the results

- **Coverage** is the cosine similarity between a respondent-topic record and
  its closest sentence in the relevant summary.
- **Mean coverage** averages that score across all retained records in a topic.
- **Bottom-decile mean** averages the scores of the lowest-covered 10% of
  records. It describes the lower tail without applying a binary threshold.
- **Low coverage** is an operational screening category defined within each
  topic as a score below the topic mean minus one standard deviation,
  $\bar{c}-\sigma$. Because the threshold is relative, the cluster distribution,
  continuous scores, bottom-decile results, and threshold-sensitivity analysis
  are more informative than the overall low-coverage percentage alone.
- **Exact-length random text** samples contiguous source-text windows matched
  to the official summary's sentence word counts. It tests performance against
  a chance floor and is not a readable-summary benchmark.
- **Mean-optimized extractive benchmark** selects complete sentences from
  training participants to maximize mean coverage under the official
  sentence-level word budgets.
- **Tail-optimized extractive benchmark** uses the same constraints while
  placing additional weight on low-covered training records.

The extractive benchmarks are evaluated only on held-out participants. They
test whether higher semantic coverage was feasible under the same space
constraint; they were not assessed as publication-ready summaries for
coherence, redundancy, policy usefulness, or human preference.

The compact results correspond primarily to Figure 5 and Tables S6--S9 of the
associated preprint. Generated participant-level outputs and embeddings are not
committed because they are recreated by the released pipeline.
