# Idea-audit protocol and amendment history

Aim: identify, without human annotation, recurring respondent ideas absent from a summary, with a detector validated on planted ground truth.

1. **Claim extraction** (`extract.py`). gpt-4o-mini, temperature 0, JSON mode, seed 0; up to 8 responses / 5,000 chars per call. Output per claim: one neutral self-contained sentence, stance (support|oppose|concern|suggestion|other), target phrase, exact supporting quote. A quote is "verified" if it is a whitespace/case/quote-normalised substring of the response; unverified quotes are flagged and used for examples only if no verified one exists. Responses dropped by the model are re-sent alone.
2. **Ideas** (`ideas.py`). Claims embedded with text-embedding-3-large; stance `other` excluded (not a position); clustering within each stance only (opposite positions cannot merge): agglomerative, cosine distance, average linkage, distance threshold 0.30 (sensitivity 0.25, 0.35). Prevalence = distinct respondents. Analysis threshold: >=10 respondents (sensitivity >=5, >=20; ideas with >=5 are stored and judged). Canonical statement: gpt-4o-mini from <=12 most central claims (one per respondent), stance preserved. Three example quotes from distinct respondents, most central first.
3. **Judges** (`judge.py`), per idea x summary:
   (a) cosine: max cosine (embedding-3-large) between statement and summary sentences (reported, not used in the rule);
   (b) NLI `cross-encoder/nli-deberta-v3-large` (loaded via sentence-transformers CrossEncoder; label order read from model config; softmax probabilities): premises = each summary sentence and the full summary (truncated to 512 tokens, hypothesis kept), hypothesis = "Respondents said that <statement>" (first letter lower-cased unless acronym/proper noun; deviation from literal "lowercased" to avoid corrupting names). Max entailment and max contradiction over premises;
   (c) LLM judge gpt-4o-mini, temp 0: expressed | partially | not_expressed | contradicted.
   **Rule**: STANCE_MISMATCH if LLM = contradicted, or (NLI contradiction >= 0.5 and entailment < 0.5) [checked first, conservative]; else MISSED if entailment < 0.5 and LLM = not_expressed; else COVERED if entailment >= 0.5 and LLM in {expressed, partially}; else UNCERTAIN. Agreement: Cohen's kappa on binary covered/not.
   **Scotland**: ideas MISSED in the section target and COVERED (same rule, sentence premises only) in the wider `full_findings` text are RELOCATED, not omissions.
4. **Validation** (`validate.py`; all on seeded samples of <=150 ideas, ideas >=10 respondents):
   (i) planted omission: for official-COVERED ideas delete every sentence that alone entails the idea (>=0.5; else the top one) and re-judge; report rate called MISSED (recall) — residual coverage by other sentences is counted as failure, so recall is conservative;
   (ii) planted inclusion: for MISSED ideas append a gpt-4o-mini paraphrase (no repeated 3-gram, up to 3 prompts; overlap reported) and report COVERED rate;
   (iii) planted flip: append an opposite-stance sentence; report false-coverage rate. Cosine contrast: cos(statement, new sentence) >= tau where tau = 10th percentile of cosines of planted paraphrases (a threshold that favours cosine by construction);
   (iv) negative control: ideas judged against the other country's official summary; expect ~all MISSED.
   Reported per judge and for the combined rule with Wilson 95% intervals.
5. **Report** (`report.py`): shares of MISSED/COVERED/STANCE_MISMATCH/UNCERTAIN/RELOCATED per topic x summary (unweighted and respondent-weighted); by prevalence bin and stance; ideas missed by all summaries vs some; `missed_ideas_<topic>.csv`; top-10 missed; sensitivity; validation.

The final rule is the v1.1 amendment below. The initial rule above is retained as development history. Automated labels and synthetic interventions have no independently human-coded gold standard. Known limits: extraction/canonicalisation use the same LLM family as one judge; planted tests are synthetic; summaries are short by design; UNCERTAIN is not counted as missed.


## v1.1 amendment (24 Oct 2026; driven by the pre-specified negative control, before any missed-idea lists were inspected)
Under v1.0, NLI contradiction >= 0.5 (with entailment < 0.5) assigned STANCE_MISMATCH ahead of MISSED. The negative control (ideas judged against an unrelated consultation's summary) returned MISSED for only 32/85 (education) and 52/126 (trust) ideas although the LLM judge said not_expressed for 84/85 and 126/126: MNLI assigns "contradiction" to unrelated statements, so it is not a stance signal. v1.1: STANCE_MISMATCH = LLM judge "contradicted"; MISSED = NLI entailment < 0.5 AND LLM "not_expressed"; COVERED unchanged; NLI contradiction is reported only. v1.0 validation outputs are kept as validation_v1.0_*.json.
