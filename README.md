# Participatory Provenance

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

## Repository structure

```text
.
├── data/
│   ├── ai-strategy-raw-data-2025.csv
│   └── data2.csv
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
│   └── run_pipeline.py
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
pip install -r src/requirements.txt
```

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
calls may incur charges.

Randomized analytical stages use fixed seeds. Because some stages depend on
external model services, a fresh run may differ slightly from the article's
reported numerical result even when the same model names and inputs are used.
The source CSV files and scripts are the released materials; generated
embeddings, credentials, and local analysis outputs are not committed.

## Data

The two CSV files contain consultation responses for the Education and Trust
policy topics. The source consultation data are available through the
[Government of Canada Open Government Portal](https://open.canada.ca/data/en/dataset/bc8cdd54-19cf-4f62-a3d3-fa4b7371d49a)
under the Open Government Licence – Canada.

## License

The analysis code is available under the MIT License. The included public data
remain subject to their source licence and terms.
