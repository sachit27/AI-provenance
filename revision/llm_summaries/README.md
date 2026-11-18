# Controlled language-model summaries

Three models summarize the same retained responses with the same order for each of three seeds. The official passage defines the sentence and word budget. Prompts use a map/reduce procedure and do not request minority-view coverage.

[PROTOCOL.md](PROTOCOL.md) records the prompts, settings and v1.1 length repair. `common.py` defines model names, seeds and protocol version. `generate.py` records final summaries and budget diagnostics, `score.py` computes coverage, and `analyze.py` compares the resulting distributions.

Use the commands in [the revision instructions](../README.md). All 36 final outputs are archived under `results/revision/llm_summaries/outputs/`. The controlled comparison includes 27 summaries for Education, Trust and Q7. Q21 is retained as an attempted experiment with budget failures.

Intermediate notes, response lists and request caches remain local. Shared prompts and inputs limit attribution of a coverage pattern to a particular summarizer. Seed-averaged participant comparisons also condition on the three generated summaries.
