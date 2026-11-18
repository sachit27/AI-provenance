"""
config.py — Central configuration for the Participatory Provenance pipeline.

All constants, paths, and summaries in one place.
"""
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ─── Paths ───────────────────────────────────────────────────────────────────

PIPELINE_DIR = Path(__file__).resolve().parent              # canadian_data/analyses/scotland_analysis/pipeline/
SCOTLAND_ORIGINAL_DIR = PIPELINE_DIR.parent                 # canadian_data/analyses/scotland_analysis/
BASE_DIR = PIPELINE_DIR.parents[2]                          # canadian_data/
OUTPUT_DIR = Path(os.environ.get(
    "PROVENANCE_OUTPUT_DIR",
    str(SCOTLAND_ORIGINAL_DIR / "analysis"),
)).expanduser().resolve()

load_dotenv(BASE_DIR / ".env")

# ─── Scottish consultation inputs (the ONLY input swap) ─────────────────────
# Official summary = the official-report section target text for each question,
# copied verbatim by analyses/scotland/code/prepare.py into summary.json.  The
# sentences are re-joined with spaces so that the unchanged pipeline sentence
# splitter (split_summary) re-derives them; build_inputs.py asserts that the
# re-split equals the stored sentence list.

SCOTLAND_DATASETS = BASE_DIR / "analyses" / "scotland" / "datasets"
INPUT_DIR = SCOTLAND_ORIGINAL_DIR / "inputs"       # written by build_inputs.py
SCOTLAND_QUESTIONS = ["q4", "q7", "q20", "q21", "q22", "q23"]

# Canadian analyst-authored atomic claims were used only by a strict-NLI
# sensitivity analysis that is not part of stages 01-08; none exist for Scotland.
ATOMIC_SUMMARY_CLAIMS = {}

# ─── Topics ──────────────────────────────────────────────────────────────────


def _load_scottish_topic(code: str) -> dict:
    with open(SCOTLAND_DATASETS / code / "summary.json", encoding="utf-8") as handle:
        sentences = json.load(handle)["sentences"]
    with open(SCOTLAND_DATASETS / code / "questions.json", encoding="utf-8") as handle:
        question = json.load(handle)
    return {
        "csv": INPUT_DIR / f"{code}.csv",
        "summary": " ".join(sentence.strip() for sentence in sentences),
        "label": question["label"],
        # Verbatim consultation question with its leading "N." numbering removed.
        "question": re.sub(r"^\s*\d+\.\s*", "", question["question"]).strip(),
        "n_official_sentences": len(sentences),
    }


TOPICS = {code: _load_scottish_topic(code) for code in SCOTLAND_QUESTIONS}

# ─── Embedding ───────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"   # OpenAI, 3072d
OPENAI_EMBEDDING_DIM = 3072
ROBUSTNESS_EMBEDDING_MODELS = [
    ("all-mpnet-base-v2", "sentence-transformers"),   # 768d, local
    ("nomic-embed-text", "ollama"),                    # 768d, local Ollama
]
EMBEDDING_BATCH_SIZE = 512   # OpenAI batch size

# ─── Clustering ──────────────────────────────────────────────────────────────

PCA_DIMS = 50
HDBSCAN_MIN_CLUSTER = 30
HDBSCAN_MIN_SAMPLES = 10
KMEANS_K_RANGE = range(4, 16)

# ─── Optimal Transport ──────────────────────────────────────────────────────

TRANSPORT_PCA_DIMS = 50           # Reduce before computing OT
BOOTSTRAP_B = 2000
PERMUTATION_N = 10000
ORPHAN_PERCENTILE = 10            # Below this percentile = orphan

# ─── Descriptive and associational analysis ────────────────────────────────

WORD_COUNT_QUINTILE_BINS = 5
ISOLATION_QUINTILE_BINS = 5

# ─── LLM Classification ────────────────────────────────────────────────────

CLASSIFICATION_MODEL = "gpt-4o-mini"
N_CLASSIFICATION_RUNS = 5
CLASSIFICATION_TEMPERATURE = 0.3
EPISTEMIC_TOP_K_EVIDENCE = 10

# ─── Topic Relevance Filtering (two-zone hybrid) ─────────────────────────────
# Zone 1: sim < RELEVANCE_REJECT  → auto-reject (clearly off-topic, no LLM)
# Zone 2: RELEVANCE_REJECT ≤ sim < RELEVANCE_ACCEPT → LLM triage (borderline)
# Zone 3: sim ≥ RELEVANCE_ACCEPT  → auto-accept (clearly on-topic, no LLM)
#
# Anchors are the verbatim CSV consultation questions + broad domain phrases so that
# minority and fringe voices that are still on-topic are never penalised.

RELEVANCE_REJECT  = 0.10   # Below this → definitely off-topic, auto-remove
RELEVANCE_ACCEPT  = 0.20   # Above this → definitely on-topic, auto-keep
RELEVANCE_LLM_MODEL = "gpt-4o-mini"   # Used only for borderline zone

# Per topic: the verbatim consultation question + three broad domain phrases
# (general domain; technology/practice; "concerns about ..." so that critical
# stances are kept), mirroring the structure of the Canadian anchors.
_SCOTLAND_DOMAIN_ANCHORS = [
    "regulation and licensing of non-surgical cosmetic procedures in Scotland",
    "botulinum toxin, dermal fillers and aesthetic treatments: practitioner training, clinics and safety",
    "concerns about the regulation, cost, access and harms of non-surgical cosmetic procedures",
]

TOPIC_RELEVANCE_ANCHORS = {
    code: [TOPICS[code]["question"], *_SCOTLAND_DOMAIN_ANCHORS]
    for code in TOPICS
}

# ─── Language Filtering ─────────────────────────────────────────────────────

FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "est", "sont", "pour", "avec",
    "dans", "sur", "une", "un", "qui", "que", "pas", "ne", "je", "nous",
    "vous", "ils", "elle", "avoir", "être", "mais", "ou", "et", "en",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
}
FRENCH_THRESHOLD = 0.08   # If >8% of words are French stopwords, classify as French
MIN_WORD_COUNT = 5

# ─── Rhetorical Register ───────────────────────────────────────────────────

HEDGE_PATTERN = (
    r'\b(perhaps|maybe|possibly|might|could|somewhat|apparently|seem[s]?|'
    r'suggest[s]?|may|arguably|conceivab|likely|unlikely|probable|'
    r'I\s+think|I\s+believe|in\s+my\s+opinion)\b'
)
ASSERTIVE_PATTERN = (
    r'\b(must|shall|need[s]?\s+to|require[ds]?|demand[s]?|'
    r'essential|critical|urgent|imperative|fundamental|'
    r'absolutely|definitely|clearly|obviously|undoubtedly|'
    r'should|obligat)\b'
)

# ─── Figures ────────────────────────────────────────────────────────────────

FIGURE_DIR = OUTPUT_DIR / "figures"
SUPP_DIR = FIGURE_DIR / "supplementary"
FIGURE_DPI = 300
SINGLE_COL_WIDTH = 3.5    # inches (Nature single column)
DOUBLE_COL_WIDTH = 7.0    # inches (Nature double column)
FONT_FAMILY = "Helvetica"
FONT_SIZE = 7
LABEL_SIZE = 8
TITLE_SIZE = 9

CLUSTER_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#e17273",
]
