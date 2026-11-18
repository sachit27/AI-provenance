# Revision analyses

Run commands from the repository root with the virtual environment activated. Install `revision/requirements.txt` after the base dependencies. The scripts retain the analysis settings used for the revised manuscript. Paths to the Canadian pipeline have been adapted to this repository's `src/` layout.

## 1. Canadian artifacts

```bash
python src/run_pipeline.py --output-root src/analysis
python revision/canada/prepare.py
```

The second command exports the retained Canadian texts, official passages and legacy embeddings for the controlled diagnostics. It also prepares the question wording used in the generation experiment.

## 2. Scottish sources and preparation

```bash
python revision/dataset_audit/scotland/fetch_public.py
python revision/dataset_audit/scotland/fetch_sources.py
python revision/dataset_audit/scotland/verify_sources.py
python revision/dataset_audit/scotland/parse_public.py
python revision/dataset_audit/scotland/verify_audit.py
python revision/scotland/code/prepare.py
```

The archive parser preserves empty and undisplayed questions separately. An independent lxml extraction checks every answer. The preparation step removes the literal editorial marker `Redacted text`, collapses whitespace, and applies a five-word minimum. The official qualitative paragraphs are fixed in `scotland/code/prepare.py`. `verify_sources.py` checks their source against the archived paragraph index before preparation.

```bash
python revision/scotland_original/pipeline/run_pipeline.py --from-raw --output-root revision/scotland_original/analysis
```

The copied pipeline applies the Canadian relevance screen with Scottish question wording and domain phrases. The resulting sample sizes are 1,021 (Q4), 674 (Q7), 575 (Q20), 756 (Q21), 549 (Q22) and 594 (Q23). The extraction before relevance screening contains 86 additional answers. Some are valid responses; the restoration analysis below checks their effect.

The official sentence budgets are 76, 8, 7, 25, 5 and 8 respectively. Q21 needs a five-word candidate minimum. The extractive selector can fall back to the full training candidate pool when its prefiltered pool cannot fill a slot. Scotland uses a Holm family of 24 mean/tail comparisons. These are the documented adaptations of the Canadian benchmark.

## 3. Controlled summaries

```bash
ollama pull qwen3.6:35b-a3b
ollama pull glm-4.7-flash:Q4_K_M
python revision/llm_summaries/generate.py --model gpt-4o-mini
python revision/llm_summaries/generate.py --model qwen
python revision/llm_summaries/generate.py --model glm
python revision/llm_summaries/score.py
python revision/llm_summaries/analyze.py
```

Run the local generation models sequentially. [Protocol v1.1](llm_summaries/PROTOCOL.md) describes the shared response order, chunking, prompts, seeds and budget repairs. Final summaries from the reported run are archived in `results/revision/llm_summaries/outputs/`. Generation creates a separate local `revision/llm_summaries/outputs/` directory. To score the archived summaries, copy that archived `outputs/` directory into `revision/llm_summaries/` before running `score.py`. Scores and comparisons still require matching participant embeddings from the primary analyses.

Q21 was attempted with all three models and seeds. Seven of nine runs missed the sentence count, so its results are diagnostic and excluded from the main controlled comparison.

## 4. Closed-answer groups and screening sensitivity

```bash
python revision/stance_groups/parse_closed.py
python revision/stance_groups/build_groups.py
python revision/stance_groups/analyze.py
python revision/scotland_original/sensitivity_restored/restore_sensitivity.py
```

[MAPPING.json](stance_groups/MAPPING.json) defines question links, group rules and test families. Stance comparisons use respondents' own closed answers. The restoration script adds all 86 answers rejected by the relevance screen and rechecks coverage and six primary group comparisons for OpenAI and MPNet. It does not rerun the cross-fitted benchmarks.

## 5. Automated idea audit

```bash
python revision/idea_audit/extract.py --topics education trust q7 q21
python revision/idea_audit/ideas.py --topics education trust q7 q21 --sensitivity-counts
python revision/idea_audit/judge.py --summaries all
python revision/idea_audit/validate.py --summaries official
python revision/idea_audit/report.py
python revision/idea_audit/stance_test.py
```

The final rule labels an idea MISSED when the language-model judge returns `not_expressed` and NLI entailment is below 0.5. `PROTOCOL.md` records the initial rule and its development-informed revision. GPT-4o-mini extracts, canonicalizes and judges ideas. A DeBERTa NLI model supplies the second judgment. Synthetic intervention results assess these labels, with no human gold standard. Recognition differs by stance category, so the reported odds ratios describe automated labels.

## 6. Controlled passage and stance diagnostics

```bash
python revision/provenance_diagnostics/run.py
```

The script uses the datasets exported in steps 1 and 2, with MPNet, OpenAI and Nomic embeddings. It constructs fixed passage-deletion, stance-reversal and condition-deletion cases. The 800 reverse-direction disagreement cases duplicate the stance comparison under symmetric cosine similarity. There are 1,651 distinct constructed cases across both countries, including 51 condition deletions. Aggregate summaries from the reported run are released; source-linked passages and respondent-level exports are generated locally.

## 7. Distribution distances

```bash
python revision/metric_sensitivity/run.py --canada education --backend openai --random-references 20 --output revision/metric_sensitivity/results/canada_education_openai
python revision/metric_sensitivity/run.py --original-root revision/scotland_original/analysis --original-pipeline revision/scotland_original/pipeline --topic q7 --backend openai --random-references 20 --output revision/metric_sensitivity/results/scotland_original_q7_openai
```

For Canada, repeat for Education and Trust with `openai` and `mpnet`. For Scotland, repeat for all six questions with both models. The Scottish Nomic runs use `--backend nomic_embed_text --random-references 0`. There are 22 archived configurations in total. Each evaluates 20, 50 and 100 principal components. The PCA basis is fitted on participant embeddings alone, which differs from the original joint participant/official-summary transport calculation.

W1 and W2 are exact transport distances. MMD² is the biased empirical V-statistic, including self-pairs, at 0.5, 1 and 2 times the median participant distance. Ranking correlations compare the official target and 20 random references within each setting. Canadian regional comparisons are saved in the JSON outputs.

## Reproduction limits

The reference results in `results/revision/` document the reported run. Large embeddings, respondent texts, intermediate claims and service caches must be regenerated. New downloads may differ from the original archive, and model services may produce different outputs. The checks supplied with this repository establish internal consistency of the release, rather than equality of every future run or independent validation of faithful representation.
