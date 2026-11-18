# Participatory provenance

Code and reference results for *Participatory provenance as representational auditing for AI-mediated public consultation*, by Sachit Mahajan, ETH Zurich.

[Preprint](https://arxiv.org/abs/2604.20711) · [Citation](CITATION.cff) · [Analysis workflow](analyses/README.md) · [Changes](CHANGELOG.md)

This study asks how semantic coverage is distributed across the people who contribute to a public consultation. It pairs each response with its closest official-summary sentence, examines the lower tail of coverage, and tests whether higher coverage is feasible within the same word limits.

The study covers two themes from Canada's 2025 AI Strategy consultation and six written questions from Scotland's 2024–25 consultation on non-surgical cosmetic procedures. It also compares summaries from three language models, examines coverage by respondents' stated positions, tests the limitations of cosine similarity, and compares distribution distances.

Semantic coverage is a screening measure. A high score can persist when a sentence reverses a position or drops a condition. The paper therefore distinguishes semantic proximity from faithful representation and reports the limitations of its automated idea judgments.

## Results

- In Canada, 5,253 retained records represent 2,861 contributors across Education and Skills and Safe AI and Public Trust. Low coverage concentrates in regions centered on criticism and distrust.
- Same-budget extraction improves held-out mean and bottom-decile coverage in both countries. The Scottish analysis uses 4,169 retained answers across six questions. These are comparisons of embedding coverage under word limits, with no prose-coherence objective.
- The Canadian official passages exceed the random-window mean with statistical significance under OpenAI embeddings. None of the Scottish passages does so. Random windows provide a chance comparison and may contain sentence fragments.
- Scottish Q7 respondents who opposed practitioner licensing received lower semantic coverage than supporters. The observed difference is −0.057, with a 95% bootstrap interval of [−0.080, −0.036] under OpenAI embeddings.
- Patterns of low coverage largely persist across model-averaged summaries. The controlled comparison includes 27 budget-compliant summaries. Nine Q21 attempts are reported separately because seven failed the sentence budget.

![Coverage, summarizer agreement and stance comparisons](docs/figures/study_results.png)

## Repository contents

| Path | Contents |
|---|---|
| `data/` | The two Canadian source CSV files |
| `src/` | Canadian analysis pipeline and figure scripts |
| `analyses/` | Scottish replication, generation prompts, stance analysis, idea audit, controlled diagnostics and distance sensitivity |
| `results/` | Canadian reference results and input hashes |
| `results/reference/` | Aggregate study results, final generated summaries and file hashes |
| `docs/figures/` | Figures from the study |
| `scripts/` | Checks that run without API credentials |

The complete study code and reference results are available at the `v2026.11` tag.

## Setup

Use Python 3.12 in a virtual environment.

```bash
git clone https://github.com/sachit27/AI-provenance.git
cd AI-provenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements-lock.txt
pip install -r analyses/requirements.txt
export OPENAI_API_KEY="your-key"
ollama pull nomic-embed-text
```

Ollama must be installed and running. Controlled generation also uses `qwen3.6:35b-a3b` and `glm-4.7-flash:Q4_K_M`. Their recorded model digests are included with the generated summaries.

Start by preparing and analyzing the Canadian consultation.

```bash
python src/run_pipeline.py --output-root src/analysis
```

The analysis scripts use `src/analysis` for the Canadian artifacts and `analyses/scotland_analysis/analysis` for Scotland. The [analysis workflow](analyses/README.md) give the download, preparation and execution order. Model calls require network access and may incur charges. Fixed analytical seeds and cached requests support repeat runs, but external services and model updates can change regenerated results.

## Reference results and checks

Aggregate outputs are included for inspection without downloading embeddings or running models. The [results guide](results/reference/README.md) maps them to the analyses and records the release scope.

```bash
python scripts/check_repository.py
python scripts/check_study.py
```

GitHub Actions runs both checks. They verify file hashes, result structure, sentence budgets and Python syntax. They do not reproduce model-dependent analyses or validate automated judgments against human annotation.

## Source data and terms

The Canadian CSVs are available through the [Government of Canada Open Government Portal](https://open.canada.ca/data/en/dataset/bc8cdd54-19cf-4f62-a3d3-fa4b7371d49a) under the Open Government Licence–Canada. Their hashes are recorded in `results/input_hashes.sha256`. Official passages are transcribed in `src/config.py` from the [ISED report](https://ised-isde.canada.ca/site/ised/en/public-consultations/engagements-canadas-next-ai-strategy-summary-inputs).

The Scottish Government published 1,809 of 2,207 responses with consent. Download scripts retrieve those pages from [Citizen Space](https://consult.gov.scot/healthcare-quality-and-improvement/regulation-of-non-surgical-cosmetic-procedures/). The response texts, quotations, identifiers, caches and embeddings are generated locally. They are excluded from this release because publication on that platform does not establish a redistribution licence for respondents' words. The official [findings report](https://www.gov.scot/isbn/9781836918271) is covered by the Open Government Licence. Selected government passages and their paragraph mapping are included.

The code is licensed under MIT. Source data and government report extracts retain their source terms. Contact Sachit Mahajan at sachit.mahajan@gess.ethz.ch.
