# Analysis workflow

Run commands from the repository root with the virtual environment activated. Install `analyses/requirements.txt` after the base dependencies. The workflow covers both consultations, controlled summaries, stance comparisons, synthetic diagnostics and distribution distances.

## 1. Canadian artifacts

```bash
python src/run_pipeline.py --output-root src/analysis
python analyses/canada/prepare.py
```

The second command exports the retained Canadian texts, official passages and legacy embeddings for the controlled diagnostics. It also prepares the question wording used in the generation experiment.

## 2. Scottish sources and preparation

```bash
python analyses/dataset_audit/scotland/fetch_public.py
python analyses/dataset_audit/scotland/fetch_sources.py
python analyses/dataset_audit/scotland/verify_sources.py
python analyses/dataset_audit/scotland/parse_public.py
python analyses/dataset_audit/scotland/verify_audit.py
python analyses/scotland/code/prepare.py
```

The archive parser preserves empty and undisplayed questions separately. An independent lxml extraction checks every answer. The preparation step removes the literal editorial marker `Redacted text`, collapses whitespace, and applies a five-word minimum. The official qualitative paragraphs are fixed in `scotland/code/prepare.py`. `verify_sources.py` checks their source against the archived paragraph index before preparation.

```bash
python analyses/scotland_analysis/pipeline/run_pipeline.py --from-raw --output-root analyses/scotland_analysis/analysis
```

The copied pipeline applies the Canadian relevance screen with Scottish question wording and domain phrases. The resulting sample sizes are 1,021 (Q4), 674 (Q7), 575 (Q20), 756 (Q21), 549 (Q22) and 594 (Q23). The extraction before relevance screening contains 86 additional answers. Some are valid responses; the restoration analysis below checks their effect.

The official sentence budgets are 76, 8, 7, 25, 5 and 8 respectively. Q21 needs a five-word candidate minimum. The extractive selector can fall back to the full training candidate pool when its prefiltered pool cannot fill a slot. Scotland uses a Holm family of 24 mean/tail comparisons. These are the documented adaptations of the Canadian benchmark.

## 3. Controlled summaries

```bash
ollama pull qwen3.6:35b-a3b
ollama pull glm-4.7-flash:Q4_K_M
python analyses/llm_summaries/generate.py --model gpt-4o-mini
python analyses/llm_summaries/generate.py --model qwen
python analyses/llm_summaries/generate.py --model glm
python analyses/llm_summaries/score.py
python analyses/llm_summaries/analyze.py
```

Run the local generation models sequentially. [Protocol v1.1](llm_summaries/PROTOCOL.md) describes the shared response order, chunking, prompts, seeds and budget repairs. Final summaries from the reported run are archived in `results/reference/llm_summaries/outputs/`. Generation creates a separate local `analyses/llm_summaries/outputs/` directory. To score the archived summaries, copy that archived `outputs/` directory into `analyses/llm_summaries/` before running `score.py`. Scores and comparisons still require matching participant embeddings from the primary analyses.

Q21 was attempted with all three models and seeds. Seven of nine runs missed the sentence count, so its results are diagnostic and excluded from the main controlled comparison.

## 4. Closed-answer groups and screening sensitivity

```bash
python analyses/stance_groups/parse_closed.py
python analyses/stance_groups/build_groups.py
python analyses/stance_groups/analyze.py
python analyses/scotland_analysis/sensitivity_restored/restore_sensitivity.py
```

[MAPPING.json](stance_groups/MAPPING.json) defines question links, group rules and test families. Stance comparisons use respondents' own closed answers. The restoration script adds all 86 answers rejected by the relevance screen and rechecks coverage and six primary group comparisons for OpenAI and MPNet. It does not rerun the cross-fitted benchmarks.

## 5. Automated idea audit

```bash
python analyses/idea_audit/extract.py --topics education trust q7 q21
python analyses/idea_audit/ideas.py --topics education trust q7 q21 --sensitivity-counts
python analyses/idea_audit/judge.py --summaries all
python analyses/idea_audit/validate.py --summaries official
python analyses/idea_audit/report.py
python analyses/idea_audit/stance_test.py
```

The final rule labels an idea MISSED when the language-model judge returns `not_expressed` and NLI entailment is below 0.5. `PROTOCOL.md` records the initial rule and its development-informed rule update. GPT-4o-mini extracts, canonicalizes and judges ideas. A DeBERTa NLI model supplies the second judgment. Synthetic intervention results assess these labels, with no human gold standard. Recognition differs by stance category, so the reported odds ratios describe automated labels.

## 6. Controlled passage and stance diagnostics

```bash
python analyses/provenance_diagnostics/run.py
```

The script uses the datasets exported in steps 1 and 2, with MPNet, OpenAI and Nomic embeddings. It constructs fixed passage-deletion, stance-reversal and condition-deletion cases. The 800 reverse-direction disagreement cases duplicate the stance comparison under symmetric cosine similarity. There are 1,651 distinct constructed cases across both countries, including 51 condition deletions. Aggregate summaries from the reported run are released; source-linked passages and respondent-level exports are generated locally.

## 7. Distribution distances

```bash
python analyses/metric_sensitivity/run.py --canada education --backend openai --random-references 20 --labels analyses/canada/datasets/education/labels.npy --output analyses/metric_sensitivity/results/canada_education_openai
python analyses/metric_sensitivity/run.py --original-root analyses/scotland_analysis/analysis --original-pipeline analyses/scotland_analysis/pipeline --topic q7 --backend openai --random-references 20 --output analyses/metric_sensitivity/results/scotland_analysis_q7_openai
```

For Canada, repeat for Education and Trust with `openai` and `mpnet`, using the matching topic’s `labels.npy` from step 1. For Scotland, repeat for all six questions with both models. The Scottish Nomic runs use `--backend nomic_embed_text --random-references 0`. There are 22 archived configurations in total. Each evaluates 20, 50 and 100 principal components. The PCA basis is fitted on participant embeddings alone, which differs from the original joint participant/official-summary transport calculation.

W1 and W2 are exact transport distances. MMD² is the biased empirical V-statistic, including self-pairs, at 0.5, 1 and 2 times the median participant distance. Ranking correlations compare the official target and 20 random references within each setting. Canadian regional comparisons are saved in the JSON outputs.

## 8. Figure 7 and supplementary tables

After running steps 1–7 from the repository root, run:

```bash
python analyses/metric_sensitivity/summarize.py
python scripts/make_fig7_tables.py --build-dir outputs/paper
```

The generator writes `fig7_extensions.png` and `supp_tables_new.tex`. The TeX file contains table fragments for incorporation into the supplement.

| Input under `analyses/` | How to create it |
|---|---|
| `scotland_analysis/scotland_analysis_summary.csv` and `scotland_analysis/analysis/q7/participant_coverage.csv` | Run the Scottish pipeline in step 2. |
| `llm_summaries/RESULTS.md` | Run `score.py` and `analyze.py` in step 3. Use the tagged summary archive to score the reported generated texts. |
| `stance_groups/respondent_groups.csv` and `stance_groups/results_all.csv` | Run the three closed-answer commands in step 4. |
| `idea_audit/RESULTS.md`, `validation_table.csv`, `summary_shares.csv`, `sensitivity.csv`, `stance_test_results.csv` and `work/<topic>/` Parquet files | Run step 5, including `validate.py`, `report.py` and `stance_test.py`. |
| `metric_sensitivity/FINDINGS.md` | Run the command above. It reads the saved distance outputs from step 7. To compare with the reported distances, use `--results-root results/reference/metric_sensitivity/results` with `summarize.py`. |
| `scotland/datasets/<question>/questions.json` | Included in the release and rebuilt in step 2. |

These commands generate response-level inputs locally. The generator contains checks against the reported analysis; a fresh model run can change classifications or scores and trigger those checks. The [reference archive](../results/reference/README.md) contains the reported aggregate outputs. For the published rendering of Figure 7, see [the study figure](../docs/figures/study_results.png).

## Reproduction limits

The reference results in `results/reference/` document the reported run. Large embeddings, respondent texts, intermediate claims and service caches must be regenerated. New downloads may differ from the original archive, and model services may produce different outputs. The checks supplied with this repository establish internal consistency of the release, rather than equality of every future run or independent validation of faithful representation.
