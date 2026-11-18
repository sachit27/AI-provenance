# Controlled passage and stance diagnostics

This module tests how maximum cosine similarity responds to constructed text changes in the Canadian and Scottish consultations. It also produces source-linked passage matches for local inspection.

```bash
python analyses/provenance_diagnostics/run.py --backends mpnet openai nomic
```

Prepare the datasets first, following [the analysis workflow](../README.md). Nomic requires a running Ollama service. Use `--ollama-url` to select a different endpoint. `--prepare-only` builds case inventories without loading embedding models. For a small trial, use `--limit-respondents` with a separate `--output` directory.

## Constructed tests

Cases are selected in a fixed hash order with seed 1729, before official-summary scores are calculated. Each topic contributes up to 100 cases per kind. Base passages contain 8–80 words, with one passage per respondent. The tests use exact passage deletion, a change from “endorses” to “rejects”, and removal of a suffix beginning with an explicit condition marker.

The reverse-direction disagreement cases duplicate the stance comparison under symmetric cosine similarity. Across both consultations there are 800 passage deletions, 800 stance reversals and 51 condition deletions. Aggregate output also retains the 800 duplicate reverse-direction cases, which are excluded from the count of distinct tests.

Exact-source retention has a cosine ceiling by construction. These tests establish sensitivity to the specified operations; they do not estimate how often a real report omits an idea or reverses a stance. Distractors come from other respondents and can share content.

## Passage matches

Eligible passages are linked to their three nearest official sentences, with offsets into the retained response. A separate Scottish full-findings target checks possible relocation. A larger target creates more opportunities for a match, so a higher score alone does not establish relocation.

The shared embedding helper handles full-text chunks and records its model identity. This differs from the primary pipeline's documented truncation policy. The controlled cases are short passages. Neither the passage scores nor the constructed tests replace response-level primary coverage.

Aggregate reference summaries are archived in `results/reference/provenance_diagnostics/`. Respondent-level matches, quotations, vectors and caches are generated locally. Seven existing tests check offsets, deterministic controls, ties, duplicate handling, constructed-vector results and cache identity.
