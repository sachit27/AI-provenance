# Automated idea audit

The audit extracts atomic claims, groups recurring ideas and judges whether each idea is expressed in a summary. It uses GPT-4o-mini and a DeBERTa NLI model. The final decision rule requires both low entailment and a language-model `not_expressed` judgment to label an idea MISSED.

[PROTOCOL.md](PROTOCOL.md) records the original design and the v1.1 rule change after the unrelated-summary control. The code implements v1.1. `stance_test.py` tests associations between automated labels and the opposing/concern category, adjusting for idea prevalence and topic. Generated-summary models use standard errors clustered by idea.

Run the commands in [the revision instructions](../README.md). The primary threshold is 0.30. Counts at 0.25 and 0.35 assess clustering sensitivity. Full judgment sensitivity requires running `ideas.py` and `judge.py` at each threshold before `report.py`.

The released aggregate outputs are under `results/revision/idea_audit/`. Extracted claims, quotations and intermediate data remain local. Synthetic interventions are conditional on ideas selected by the automated procedure. They show differential recognition across stance categories and do not calibrate error rates on real summaries. The reported associations therefore concern labels produced by this pipeline.
