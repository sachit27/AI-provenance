#!/usr/bin/env python3
"""
01_preprocess.py — Data cleaning and Nomic robustness embeddings.

Cleans both consultation topics and creates the participant/official-sentence
Nomic embeddings used only for model robustness.  The combined OpenAI and
MPNet analysis corpora are constructed once, after preprocessing, by
02_prepare_benchmarks.py --prepare-only.

Usage:
    python 01_preprocess.py
"""
import json
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    NumpyEncoder,
    TOPICS, OUTPUT_DIR,
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM,
    ROBUSTNESS_EMBEDDING_MODELS,
    FRENCH_STOPWORDS, FRENCH_THRESHOLD, MIN_WORD_COUNT,
    HEDGE_PATTERN, ASSERTIVE_PATTERN,
    TOPIC_RELEVANCE_ANCHORS, RELEVANCE_REJECT, RELEVANCE_ACCEPT,
    RELEVANCE_LLM_MODEL,
)


# ─── Language Detection ─────────────────────────────────────────────────────

def is_french(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 3:
        return False
    french_count = sum(1 for w in words if w in FRENCH_STOPWORDS)
    return (french_count / len(words)) > FRENCH_THRESHOLD


# ─── Rhetorical Register ───────────────────────────────────────────────────

HEDGE_RE = re.compile(HEDGE_PATTERN, re.IGNORECASE)
ASSERTIVE_RE = re.compile(ASSERTIVE_PATTERN, re.IGNORECASE)


def classify_register(text: str) -> dict:
    words = text.split()
    n_words = max(1, len(words))
    n_hedge = len(HEDGE_RE.findall(text))
    n_assert = len(ASSERTIVE_RE.findall(text))
    hedge_rate = n_hedge / n_words
    assert_rate = n_assert / n_words

    if assert_rate > hedge_rate * 1.5:
        register = "assertive"
    elif hedge_rate > assert_rate * 1.5:
        register = "hedged"
    else:
        register = "mixed"

    return {
        "hedge_rate": round(hedge_rate, 6),
        "assertiveness_rate": round(assert_rate, 6),
        "register": register,
    }


# ─── Ollama Embedding ──────────────────────────────────────────────────────

def embed_ollama(texts: list[str], model_name: str) -> np.ndarray:
    import subprocess
    embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc=f"  Ollama/{model_name}"):
        batch = texts[i:i + 32]
        batch = [t[:4000] for t in batch]  # Ollama context limit
        payload = json.dumps({"model": model_name, "input": batch})
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/embed", "-d", payload],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        embeddings.extend(data["embeddings"])
    return np.array(embeddings, dtype=np.float32)


# ─── Summary Sentence Splitting ────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    # Split on period followed by space + capital letter, or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201C])', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def count_sentences(text: str) -> int:
    """Count non-empty sentence-like spans without a terminal empty split."""
    spans = re.split(r'(?<=[.!?])\s+|[\r\n]+', text.strip())
    return max(1, sum(bool(span.strip()) for span in spans))


# ─── Main ──────────────────────────────────────────────────────────────────

def process_topic(topic_name: str, topic_config: dict):
    print(f"\n{'='*60}")
    print(f"Processing: {topic_config['label']}")
    print(f"{'='*60}")

    topic_dir = OUTPUT_DIR / topic_name
    topic_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df = pd.read_csv(topic_config["csv"], low_memory=False)
    df["_source_row"] = np.arange(len(df), dtype=int)
    audit = pd.DataFrame({
        "source_row": df["_source_row"],
        "Internal ID": df["Internal ID"],
        "exclusion_reason": "",
        "relevance_score": np.nan,
        "relevance_zone": "not_evaluated",
        "llm_relevance_verdict": "not_evaluated",
        "final_included": False,
    }).set_index("source_row")
    cols = [c for c in df.columns if c not in {"Internal ID", "_source_row"}]
    print(f"  Raw rows: {len(df)}")

    # Concatenate all question responses per respondent
    df["text"] = df[cols].apply(
        lambda row: " ".join(str(v) for v in row if pd.notna(v) and str(v).strip()),
        axis=1,
    ).str.strip()

    # Remove empty
    n_before = len(df)
    empty_mask = df["text"].str.len() == 0
    audit.loc[df.loc[empty_mask, "_source_row"], "exclusion_reason"] = "empty"
    df = df[~empty_mask].copy()
    n_empty = n_before - len(df)
    print(f"  Removed empty: {n_empty}")

    # Language detection
    df["is_french"] = df["text"].apply(is_french)
    n_french = df["is_french"].sum()
    french_ids = df[df["is_french"]]["Internal ID"].tolist()
    audit.loc[df.loc[df["is_french"], "_source_row"], "exclusion_reason"] = "french_heuristic"
    df = df[~df["is_french"]].copy()
    print(f"  Removed French: {n_french}")

    # Word count
    df["word_count"] = df["text"].str.split().str.len()

    # Remove very short
    n_short = (df["word_count"] < MIN_WORD_COUNT).sum()
    short_mask = df["word_count"] < MIN_WORD_COUNT
    audit.loc[df.loc[short_mask, "_source_row"], "exclusion_reason"] = "below_minimum_word_count"
    df = df[~short_mask].copy()
    print(f"  Removed short (<{MIN_WORD_COUNT} words): {n_short}")

    # Spam filter: remove Lorem ipsum placeholder text, repetitive gibberish,
    # and responses with excessive character/word repetition.
    def is_spam(text: str) -> bool:
        t_lower = text.lower()
        # Lorem ipsum placeholder content
        lorem_markers = ["lorem ipsum", "amet consectetur", "adipiscing elit",
                         "sed do eiusmod", "labore et dolore", "facilisis efficitur",
                         "vestibulum sodales", "pellentesque habitant"]
        if any(m in t_lower for m in lorem_markers):
            return True
        # Repeated single word (e.g. "test test test test")
        words = t_lower.split()
        if len(words) >= 4:
            most_common_freq = max(words.count(w) for w in set(words))
            if most_common_freq / len(words) > 0.6:
                return True
        # Keyboard mash: > 40% non-alpha chars relative to word chars
        alpha = sum(c.isalpha() for c in text)
        if alpha > 0 and (len(text) - alpha) / len(text) > 0.6:
            return True
        return False

    n_before_spam = len(df)
    spam_mask = df["text"].apply(is_spam)
    audit.loc[df.loc[spam_mask, "_source_row"], "exclusion_reason"] = "spam_or_placeholder"
    df = df[~spam_mask].copy()
    n_spam = n_before_spam - len(df)
    print(f"  Removed spam/invalid: {n_spam}")

    # ── Two-zone topic relevance filter ──────────────────────────────────
    # Stage 1 (embedding): fast local cosine similarity against consultation
    #   anchors. Splits responses into auto-reject / borderline / auto-accept.
    # Stage 2 (LLM triage): only borderline responses go to gpt-4o-mini.
    # Protects minority voices — even strong fringe opposition is kept as long
    # as it references the consultation domain.
    n_removed = 0
    anchors = TOPIC_RELEVANCE_ANCHORS.get(topic_name, [])
    if anchors:
        print(f"\n  [Relevance] Stage 1 — embedding similarity...")
        from sentence_transformers import SentenceTransformer
        df = df.reset_index(drop=True)
        rel_model = SentenceTransformer("all-mpnet-base-v2")
        anchor_embs = rel_model.encode(anchors, convert_to_numpy=True,
                                        show_progress_bar=False)
        text_embs = rel_model.encode(df["text"].tolist(), convert_to_numpy=True,
                                      batch_size=64, show_progress_bar=True)
        anchor_embs /= (np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-10)
        text_embs   /= (np.linalg.norm(text_embs,   axis=1, keepdims=True) + 1e-10)
        max_sim = (text_embs @ anchor_embs.T).max(axis=1)  # (n,)
        source_rows = df["_source_row"].to_numpy()
        audit.loc[source_rows, "relevance_score"] = max_sim

        auto_reject  = max_sim < RELEVANCE_REJECT
        borderline   = (max_sim >= RELEVANCE_REJECT) & (max_sim < RELEVANCE_ACCEPT)
        auto_accept  = max_sim >= RELEVANCE_ACCEPT
        audit.loc[source_rows[auto_reject], "relevance_zone"] = "auto_reject"
        audit.loc[source_rows[borderline], "relevance_zone"] = "borderline_llm"
        audit.loc[source_rows[auto_accept], "relevance_zone"] = "auto_accept"

        print(f"    Auto-accept  (sim ≥ {RELEVANCE_ACCEPT}): {int(auto_accept.sum())}")
        print(f"    Borderline   ({RELEVANCE_REJECT}–{RELEVANCE_ACCEPT}): {int(borderline.sum())}")
        print(f"    Auto-reject  (sim < {RELEVANCE_REJECT}): {int(auto_reject.sum())}")

        # Stage 2: LLM triage for borderline only
        borderline_idx = np.where(borderline)[0]
        llm_keep = np.ones(len(borderline_idx), dtype=bool)  # default: keep

        if len(borderline_idx) > 0:
            print(f"\n  [Relevance] Stage 2 — LLM triage for {len(borderline_idx)} borderline responses...")
            from openai import OpenAI as _OAI
            _llm_client = _OAI(api_key=OPENAI_API_KEY)
            TRIAGE_SYSTEM = (
                "You are a relevance filter for a Canadian government AI strategy consultation. "
                "Your only job is to classify whether a response is on-topic for this consultation. "
                "The consultation covers: artificial intelligence policy, AI education, digital skills, "
                "AI safety, AI regulation, public trust in AI, and technology governance in Canada. "
                "A response is VALID if it expresses any opinion, concern, question, or idea related "
                "to these topics — even if strongly critical, unconventional, or off-putting. "
                "A response is NOISE only if it has zero connection to AI, technology, education, "
                "or government policy (e.g., unrelated personal statements, gibberish). "
                "Respond with ONLY the single word: Valid  or  Noise"
            )
            for i, idx in enumerate(borderline_idx):
                text_snippet = df.loc[idx, "text"][:400]
                try:
                    resp = _llm_client.chat.completions.create(
                        model=RELEVANCE_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": TRIAGE_SYSTEM},
                            {"role": "user",   "content": f'Response: "{text_snippet}"'},
                        ],
                        temperature=0,
                        max_tokens=5,
                    )
                    verdict = resp.choices[0].message.content.strip().lower()
                    llm_keep[i] = ("valid" in verdict)
                except Exception:
                    pass  # On error: default keep (conservative)
                if (i + 1) % 50 == 0:
                    print(f"    ... triaged {i+1}/{len(borderline_idx)}")
                time.sleep(0.05)

            audit.loc[source_rows[borderline_idx], "llm_relevance_verdict"] = np.where(
                llm_keep, "valid", "noise"
            )
            n_llm_rejected = int((~llm_keep).sum())
            print(f"    LLM rejected: {n_llm_rejected} / {len(borderline_idx)}")

        # Build final keep mask
        keep_mask = auto_accept.copy()
        for i, idx in enumerate(borderline_idx):
            keep_mask[idx] = llm_keep[i]
        # auto_reject stays False

        rejected_rows = source_rows[~keep_mask]
        audit.loc[rejected_rows, "exclusion_reason"] = np.where(
            auto_reject[~keep_mask], "relevance_auto_reject", "relevance_llm_reject"
        )
        n_removed = int((~keep_mask).sum())
        df = df[keep_mask].copy()
        print(f"  Total removed (off-topic): {n_removed} responses")

    # Assign IDs
    df["id"] = "P" + df["Internal ID"].astype(str)
    df = df.reset_index(drop=True)

    # Sentence count
    df["sentence_count"] = df["text"].apply(count_sentences)

    # Rhetorical register
    rhet = df["text"].apply(classify_register)
    df["hedge_rate"] = rhet.apply(lambda x: x["hedge_rate"])
    df["assertiveness_rate"] = rhet.apply(lambda x: x["assertiveness_rate"])
    df["register"] = rhet.apply(lambda x: x["register"])

    n = len(df)
    print(f"  Final participants: {n}")
    print(f"  Word count: median={df['word_count'].median():.0f}, "
          f"mean={df['word_count'].mean():.1f}, max={df['word_count'].max()}")
    print(f"  Register: {dict(df['register'].value_counts())}")

    # ── Summary sentences ────────────────────────────────────────────────
    summary = topic_config["summary"]
    summary_sentences = split_sentences(summary)
    print(f"\n  Summary sentences: {len(summary_sentences)}")
    for i, s in enumerate(summary_sentences):
        print(f"    S{i}: {s[:80]}...")

    # ── Nomic robustness embeddings ─────────────────────────────────────
    texts = df["text"].tolist()
    all_texts = texts + summary_sentences
    nomic_model = "nomic-embed-text"
    print(f"\n  Embedding core robustness rows with {nomic_model}...")
    nomic_embeddings = embed_ollama(all_texts, nomic_model)

    # ── Save ─────────────────────────────────────────────────────────────
    save_cols = ["id", "Internal ID", "text", "word_count", "sentence_count",
                 "hedge_rate", "assertiveness_rate", "register"]
    df[save_cols].to_parquet(topic_dir / "clean.parquet", index=False)

    # Row-level preprocessing audit: every exclusion is traceable by source row
    # and respondent ID. Participant text is omitted to avoid duplicating raw data.
    audit.loc[df["_source_row"], "final_included"] = True
    audit.reset_index().to_parquet(topic_dir / "filter_audit.parquet", index=False)

    np.savez_compressed(
        topic_dir / "embeddings_nomic_embed_text.npz",
        input_embeddings=nomic_embeddings[:n],
        summary_embeddings=nomic_embeddings[n:],
    )

    # Metadata
    meta = {
        "topic": topic_name,
        "label": topic_config["label"],
        "n_raw": int(n_before),
        "n_empty": int(n_empty),
        "n_french": int(n_french),
        "n_short": int(n_short),
        "n_spam": int(n_spam),
        "n_off_topic": int(n_removed) if anchors else 0,
        "n_final": int(n),
        "n_summary_sentences": len(summary_sentences),
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "openai_embedding_dim": OPENAI_EMBEDDING_DIM,
        "robustness_models": [m[0] for m in ROBUSTNESS_EMBEDDING_MODELS],
        "word_count_stats": {
            "median": float(df["word_count"].median()),
            "mean": float(df["word_count"].mean()),
            "std": float(df["word_count"].std()),
            "max": int(df["word_count"].max()),
        },
        "register_distribution": {k: int(v) for k, v in df["register"].value_counts().items()},
    }
    with open(topic_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    print(f"\n  \u2713 Saved to {topic_dir}/")
    return n


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = {}
    for topic_name, topic_config in TOPICS.items():
        n = process_topic(topic_name, topic_config)
        totals[topic_name] = n

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    for t, n in totals.items():
        print(f"  {t}: {n} participants")


if __name__ == "__main__":
    main()
