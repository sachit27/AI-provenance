# Scottish replication

This copy of the Canadian pipeline uses the Scottish inputs prepared by `revision/scotland/code/prepare.py`. See [the execution instructions](../README.md) for the download and analysis order.

The changes concern input paths, relevance anchors and domain wording, variable sentence budgets, the Q21 five-word candidate minimum, training-candidate fallback, pairwise question overlap, and the 24-test Holm family. Core coverage, transport and association calculations retain the original method.

Outputs are generated under `analysis/`. Reference aggregate results are archived in `results/revision/scotland_original/` at the repository root. `sensitivity_restored/restore_sensitivity.py` checks coverage and stance comparisons after restoring answers rejected by the relevance screen.
