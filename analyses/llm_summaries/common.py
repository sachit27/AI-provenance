"""Shared helpers for the controlled multi-LLM summary experiment (see PROTOCOL.md)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # canadian_data/
OUT_DIR = HERE / "outputs"
SCORE_DIR = HERE / "scores"
CACHE_DB = HERE / "cache.sqlite"

PROTOCOL_VERSION = "1.1"
TEMPERATURE = 0.7
NUM_CTX = 32768
CHUNK_TOKENS = 20000          # approx tokens (chars/4) of participant text per map chunk
DEFAULT_SEEDS = [1729, 1730, 1731]
ALL_TOPICS = ["education", "trust", "q7", "q21"]

MODELS = {
    "gpt-4o-mini": dict(kind="openai", name="gpt-4o-mini", slug="gpt-4o-mini"),
    "qwen": dict(kind="ollama", name="qwen3.6:35b-a3b", slug="qwen3.6-35b-a3b"),
    "glm": dict(kind="ollama", name="glm-4.7-flash:Q4_K_M", slug="glm-4.7-flash"),
}
SLUGS = [m["slug"] for m in MODELS.values()]

CASES = {
    "education": "canada", "trust": "canada", "q7": "scotland", "q21": "scotland",
}
ANALYSIS_ROOT = {
    "canada": ROOT / "src" / "analysis",
    "scotland": ROOT / "analyses" / "scotland_analysis" / "analysis",
}
PIPELINE_MODULE_DIR = {
    "canada": ROOT / "src",
    "scotland": ROOT / "analyses" / "scotland_analysis" / "pipeline",
}
EMB_MODELS = ["openai", "mpnet"]

ID_CANDIDATES = ["respondent_id", "Internal ID", "id", "response_id", "participant_id"]
TEXT_CANDIDATES = ["text", "response", "response_text", "answer"]


def split_summary(text: str) -> list[str]:
    """Identical to src/analysis_io.split_summary."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text.strip())
    return [part.strip() for part in parts if part.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def sha256(obj) -> str:
    if not isinstance(obj, (str, bytes)):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    if isinstance(obj, str):
        obj = obj.encode("utf-8")
    return hashlib.sha256(obj).hexdigest()


def pick_col(columns, candidates, override=None, what="column"):
    if override:
        if override not in columns:
            raise KeyError(f"{what} {override!r} not in {list(columns)}")
        return override
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"cannot auto-detect {what}; columns are {list(columns)}")


def load_topic(topic: str, id_col=None, text_col=None, allow_records_fallback=False) -> dict:
    """Return dict(question, ids, texts, budget_sentences, budget_words, case)."""
    import pandas as pd
    case = CASES[topic]
    clean = ANALYSIS_ROOT[case] / topic / "clean.parquet"
    if case == "canada":
        spec = importlib.util.spec_from_file_location("canada_config", ROOT / "src" / "config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        official = cfg.TOPICS[topic]["summary"]
        b_sent, b_words = len(split_summary(official)), word_count(official)
        q = json.loads((ROOT / "analyses" / "canada" / "datasets" / topic / "questions.json").read_text())
        question = " | ".join(q["questions"])
    else:
        base = ROOT / "analyses" / "scotland" / "datasets" / topic
        ds = json.loads((base / "dataset.json").read_text())
        b_sent, b_words = int(ds["summary_sentences"]), int(ds["summary_words"])
        question = json.loads((base / "questions.json").read_text())["question"]
    if clean.exists():
        df = pd.read_parquet(clean)
        ic = pick_col(df.columns, ID_CANDIDATES, id_col, "id column")
        tc = pick_col(df.columns, TEXT_CANDIDATES, text_col, "text column")
        ids = df[ic].astype(str).tolist()
        texts = df[tc].astype(str).tolist()
    elif allow_records_fallback and case == "scotland":
        print(f"WARNING: {clean} missing; using analyses/scotland/datasets/{topic}/records.jsonl", file=sys.stderr)
        rows = [json.loads(l) for l in (ROOT / "analyses" / "scotland" / "datasets" / topic / "records.jsonl").read_text().splitlines() if l.strip()]
        ids = [str(r["respondent_id"]) for r in rows]
        texts = [r["text"] for r in rows]
    else:
        raise FileNotFoundError(f"{clean} not found (Scotland port not ready? use --allow-records-fallback)")
    return dict(topic=topic, case=case, question=question, ids=ids, texts=texts,
                budget_sentences=b_sent, budget_words=b_words)
