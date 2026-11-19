# Controlled multi-LLM summarisation protocol (version 1.1)

Purpose: test whether coverage patterns of a consultation summary come from the summariser, from human editorial
choices (official summary), or from the dataset, by producing summaries from the same inputs under one identical protocol.

**Models**: gpt-4o-mini (OpenAI API); qwen3.6:35b-a3b and glm-4.7-flash:Q4_K_M (Ollama, `/api/chat`, `think:false`,
`stream:false`). Ollama digests from `/api/tags` are stored in each `summary.json` (`model_digest`); for OpenAI the
resolved model string of the final call is stored.
**Topics**: Canada `education`, `trust`; Scotland `q7`, `q21`. **Seeds**: 1729, 1730, 1731. **Temperature**: 0.7 for all calls.
Ollama options: `num_ctx=32768`, `seed`, `temperature`, `num_predict` (map 700, merge 900, final max(500, 3×word budget)).
OpenAI options: `temperature`, `seed`, `max_tokens` (same values). (OpenAI seed is best-effort determinism only.)

**Inputs**: retained participant texts (`clean.parquet` of the original analysis for the case). Budget: Canada = sentence and
whitespace-word count of the official summary in `pipeline/config.py` (6 sentences); Scotland = `summary_sentences`/`summary_words`
in `analyses/scotland/datasets/<q>/dataset.json`.

**Procedure** (per topic, model, seed):
1. Shuffle responses with `random.Random(seed)` (identical order for all models).
2. Chunk sequentially by approximate tokens (chars/4) with a cap of 20,000 tokens of response text per chunk (identical chunks for all models;
   fits num_ctx 32768 with prompt and output). A single response longer than the cap is truncated (reported).
3. Map: per chunk, system "You are a careful policy analyst working for a government department."; user: "You are analysing public
   consultation responses to the question: <question>. Write concise analytic notes (max 250 words) on the views expressed in these
   responses." followed by the numbered responses. No mention of minorities, diversity or representation.
4. Reduce: while the concatenated notes exceed ~20,000 tokens, merge them in groups (<=10,000 tokens each, max 400 words per merged note).
5. Final: "Write a summary of what respondents said in exactly N sentences and at most W words. Use plain prose: no lists, headings or
   bullet points. Output only the summary."
6. Validate with `analysis_io.split_summary`'s regex (sentence count must equal N) and whitespace word count (<= W). On failure, up to 3 retries
   that append the failed answer and explicit feedback (counts vs requirement). If still failing, the attempt with the smallest
   violation (|sentence error|, then excess words) is kept and `budget_compliant=false`.
7. `<think>...</think>` blocks are stripped; markdown `**` removed; whitespace collapsed.

**Caching**: every LLM call is stored in `cache.sqlite` keyed by sha256(model, messages, seed, options, think=false), so reruns resume.

**Scoring** (score.py): sentences are embedded with the same models/settings as the original pipeline (OpenAI `text-embedding-3-large`
with text truncated to 8000 chars, batch 512; `all-mpnet-base-v2` via sentence-transformers, default `encode`), and coverage = max
cosine similarity to any summary sentence, computed exactly as `04_transport.compute_coverage`. Participant and official-summary
embeddings are loaded through the original `analysis_io.embedding_blocks` of the respective case.

**Analysis** (analyze.py): mean coverage, bottom-decile mean (ceil(10%) lowest), low-coverage rate at the original frozen threshold
(`frozen_exclusion_threshold` in `benchmark_<model>.json` = official mean - 1 SD; recomputed if absent) plus a relative-threshold variant,
Gini; participant-bootstrap paired CIs vs official; Spearman between summarisers; Jaccard of bottom-decile sets vs random baseline;
per-region (`cluster_id` in `clustered.parquet`) mean-coverage ranks and Kendall's W with permutation p-values.

## Changes from version 1.0 (24 Oct 2026, before any coverage scoring)
Pilot outputs showed correct sentence counts but occasional word-budget overshoot (1–15 words) after the three chained retries. v1.1 adds, identically for every model and only when those retries fail, up to three fresh "condense" requests on the best attempt (target ≈ budget − 10 words, never above budget, same sentence count, no new content; seeds seed+1000·k). `condense_repair_used` is recorded per summary. All earlier calls are reused from the cache, so v1.0-compliant summaries are unchanged. The decision was made from length diagnostics only; no coverage scores existed.

The condense request states the current and required sentence and word counts and may merge, split or shorten sentences, so it also repairs sentence-count violations (seen for the 25-sentence q21 budget).
