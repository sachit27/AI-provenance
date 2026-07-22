# Participatory Provenance

[Paper on arXiv](https://arxiv.org/abs/2604.20711) ·
[DOI](https://doi.org/10.48550/arXiv.2604.20711) ·
[How to cite](CITATION.cff)

This repository contains the data and analysis pipeline for studying how well
official summaries represent the range of views submitted to a public
consultation. The case study examines two policy areas from Canada's 2025 AI
Strategy consultation: **Education and Skills** and **Safe AI and Public
Trust**.

The analysis measures semantic coverage at the level of individual responses,
examines where low coverage concentrates, compares patterns across topics and
embedding models, and evaluates official summaries against length-matched
random text and leakage-controlled extractive benchmarks. In the cross-fitted
benchmark, fold assignment is independent of the full-corpus clusters and
candidate filtering uses training records only. Low coverage is an operational
screening measure; it does not imply deliberate exclusion.

## Associated manuscript

- **Title:** Participatory provenance as representational auditing for
  AI-mediated public consultation
- **Author:** Sachit Mahajan, ETH Zurich
- **Contact:** sachit.mahajan@gess.ethz.ch
- **Preprint:** arXiv:2604.20711, version 3 (20 July 2026)
- **DOI:** [10.48550/arXiv.2604.20711](https://doi.org/10.48550/arXiv.2604.20711)
- **Journal status:** Under review
- **Submission version:** `patterns-submission-v1`

The submission tag identifies the code, data, and compact reference results
provided for editorial and peer review. Later development can continue on
`main` without changing that submitted version.

## Main findings

- The official summaries had higher observed mean semantic coverage than
  exact-length random text in both topics and under both primary embedding
  models. The chance comparison was statistically significant with
  `text-embedding-3-large`, but not with `all-mpnet-base-v2`. The random-text
  comparison is a chance floor, not a benchmark for readable summaries.
- Low coverage was concentrated in particular semantic regions. It reached
  80.6% in the Education region centered on criticism of educational
  technology and 88.1% in the Trust region centered on distrust of technology
  and oversight. No records met the same operational threshold in the
  best-covered regions centered on workforce skills and building public trust.
- Among respondents represented in both topics, low-coverage status was
  associated across topics (odds ratio 6.54). This is a descriptive association,
  not a causal estimate.
- Same-budget, cross-fitted extractive benchmarks improved held-out mean
  coverage by 0.057--0.090 and bottom-decile coverage by 0.047--0.081. These
  benchmarks test feasible semantic coverage under the stated constraints;
  they are not proposed as publication-ready summaries.

![Participatory provenance workflow](docs/figures/participatory_provenance_overview.png)

The framework links each response to its closest official-summary sentence,
then examines the distribution and location of low semantic coverage. The UMAP
panels are visualizations of the embedding space and are not used to calculate
coverage or statistical tests.

## Repository structure

```text
.
├── data/
│   ├── ai-strategy-raw-data-2025.csv
│   └── data2.csv
├── docs/
│   └── figures/
│       ├── coverage_benchmarks.png
│       └── participatory_provenance_overview.png
├── .github/
│   └── workflows/
│       └── repository-checks.yml
├── scripts/
│   └── check_repository.py
├── src/
│   ├── 01_preprocess.py
│   ├── 02_prepare_benchmarks.py
│   ├── 03_topology.py
│   ├── 04_transport.py
│   ├── 05_associations.py
│   ├── 06_cross_topic.py
│   ├── 07_crossfit_benchmarks.py
│   ├── 08_summarize_benchmarks.py
│   ├── 09_figures.py
│   ├── verify_sentence_assignment.py
│   ├── analysis_io.py
│   ├── config.py
│   ├── requirements.txt
│   ├── requirements-lock.txt
│   └── run_pipeline.py
├── results/
│   ├── crossfit_confirmatory_summary.json
│   ├── input_hashes.sha256
│   └── README.md
├── CITATION.cff
├── LICENSE
└── README.md
```

## Analysis workflow

```mermaid
flowchart LR
    A[Consultation CSV files] --> B[Clean and filter responses]
    B --> C[Create OpenAI, MPNet and Nomic embeddings]
    C --> D[Select and assess semantic partitions]
    D --> E[Measure coverage, inequality and transport distance]
    E --> F[Descriptive associations and robustness checks]
    F --> G[Chance and cross-fitted same-budget benchmarks]
    G --> H[Tables, statistical summaries and figures]
```

## Setup

Python 3.11 or later is recommended.

```bash
git clone https://github.com/sachit27/AI-provenance.git
cd AI-provenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements-lock.txt
```

The pinned file records the Python 3.12 submission environment and has been
installed and import-tested in a clean environment. `src/requirements.txt`
retains compatible minimum versions for later development.

The complete raw-data run uses:

- `text-embedding-3-large` for the OpenAI embedding analysis;
- `gpt-4o-mini` for borderline relevance adjudication and initial assistance
  with cluster labels;
- `all-mpnet-base-v2` as a local embedding-model robustness check;
- `nomic-embed-text` through a local [Ollama](https://ollama.com/) service as
  an additional robustness check.

The repository contains no credentials. Set the OpenAI API key in the process
environment without committing it:

```bash
export OPENAI_API_KEY="your-key"
```

Install Ollama separately and make the Nomic model available:

```bash
ollama pull nomic-embed-text
```

## Run the analysis

From the repository root:

```bash
python src/run_pipeline.py --output-root outputs/analysis
```

The output directory will contain cleaned records, embeddings, cluster and
coverage results, robustness analyses, benchmark summaries, sentence-level
assignment artifacts, and figures. The runner generates and structurally verifies
`sentence_assignment.json` for both topics with
`src/verify_sentence_assignment.py`. To verify an existing output directory
without rerunning the pipeline, use:

```bash
PROVENANCE_OUTPUT_DIR=outputs/analysis python src/verify_sentence_assignment.py
```

The pipeline can require substantial memory and compute time, and OpenAI API
calls may incur charges. Exact runtime and cost depend on hardware, service
throughput, and whether embeddings are generated afresh.

Randomized analytical stages use fixed seeds. Because some stages depend on
external model services, a fresh run may differ slightly from the article's
reported numerical result even when the same model names and inputs are used.
The source CSV files and scripts are the released materials. Apart from the
compact reference summary below, generated embeddings, credentials, and local
analysis outputs are not committed.

## Reference results

`results/crossfit_confirmatory_summary.json` is the compact output produced by
the final confirmatory benchmark summarization stage. It is supplied so that a
reviewer can compare a new run with the submitted analysis without downloading
large embedding files. It is a reference result, not an input to the pipeline.

| Topic | Embedding model | Mean-coverage gain (95% CI) | Bottom-decile gain (95% CI) |
|---|---|---:|---:|
| Education | text-embedding-3-large | 0.065 [0.063, 0.067] | 0.047 [0.041, 0.053] |
| Education | all-mpnet-base-v2 | 0.078 [0.076, 0.081] | 0.061 [0.054, 0.069] |
| Trust | text-embedding-3-large | 0.057 [0.055, 0.059] | 0.056 [0.049, 0.063] |
| Trust | all-mpnet-base-v2 | 0.090 [0.088, 0.093] | 0.081 [0.072, 0.089] |

All eight primary paired-randomization comparisons have Holm-adjusted
`p = 0.0016`. The JSON file retains the unrounded estimates, intervals,
chance comparisons, repeated-partition ranges, and worst-region diagnostics.
Definitions and interpretation notes for these outputs are provided in
[`results/README.md`](results/README.md).

![Held-out same-budget benchmark results](docs/figures/coverage_benchmarks.png)

The benchmark comparisons use held-out records and the official summary's
sentence-level word budgets. Error bars show conditional participant-level
bootstrap intervals. See the paper and supplement for the complete design and
inferential specification.

## Repository checks

The lightweight check verifies the two public data-file hashes, the structure
of the compact confirmatory results, the eight-test primary family, and Python
syntax without requiring API credentials:

```bash
python scripts/check_repository.py
```

GitHub Actions runs the same credential-free check on pushes and pull requests.
It does not rerun the model-dependent analysis.

## Data

The two CSV files contain consultation responses for the Education and Trust
policy topics. Each source file has 11,383 respondent rows before the documented
topic-specific filtering and concatenation steps.

| File | Topic | SHA-256 |
|---|---|---|
| `data/ai-strategy-raw-data-2025.csv` | Education and Skills | `33316259d7675e90b97223bdfcc0dcd06e613e54b48c2fcefb51604e08d458a7` |
| `data/data2.csv` | Safe AI and Public Trust | `d22c3b21bb42a53d68cacd6dfe308480e04e18df9ecf9996422839fd7cc128c1` |

The checksums are also available in `results/input_hashes.sha256`. The source
consultation data are available through the
[Government of Canada Open Government Portal](https://open.canada.ca/data/en/dataset/bc8cdd54-19cf-4f62-a3d3-fa4b7371d49a)
under the Open Government Licence - Canada. The six official summary sentences
per topic were transcribed from the Government of Canada report
[Engagements on Canada's Next AI Strategy: Summary of Inputs](https://ised-isde.canada.ca/site/ised/en/public-consultations/engagements-canadas-next-ai-strategy-summary-inputs)
and are defined verbatim in `src/config.py`.

## License

The analysis code is available under the MIT License. The included public data
remain subject to their source licence and terms.

## Citation

Please cite the associated preprint when using this code or data:

> Mahajan, S. (2026). Participatory provenance as representational auditing
> for AI-mediated public consultation. arXiv:2604.20711.
> https://doi.org/10.48550/arXiv.2604.20711
