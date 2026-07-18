"""
config.py — Central configuration for the Participatory Provenance pipeline.

All constants, paths, and summaries in one place.
"""
import json
import os
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get(
    "PROVENANCE_OUTPUT_DIR",
    str(PIPELINE_DIR / "analysis"),
)).expanduser().resolve()

load_dotenv(BASE_DIR / ".env")

# ─── Official Summaries (verbatim from AiStrategyReport_EN.pdf, page 8) ─────

EDUCATION_SUMMARY = (
    "The respondents had a clear view: Canada needs a dual approach to skills "
    "development involving broad digital literacy for all citizens and advanced "
    "AI expertise for specialized roles. Respondents called for integrating AI "
    "education across K\u201312 and post-secondary curricula, offering short, "
    "stackable credentials, and investing in lifelong learning. Public "
    "infrastructure must ensure equitable access to these opportunities. "
    "Critical thinking, ethical reasoning and interdisciplinary skills were "
    "emphasized over narrow technical training. Respondents urged Canada to "
    "develop a national AI literacy strategy, workplace training programs and "
    "public campaigns to demystify AI. Concerns include job displacement, "
    "environmental harm and cognitive dependency, reinforcing the need for "
    "human-centred education."
)

TRUST_SUMMARY = (
    "Public trust in AI hinges on transparency, accountability and robust "
    "governance. Respondents called for risk-based certification standards, "
    "independent audits, and clear disclosures about AI use. Ethical "
    "guidelines and oversight bodies are seen as essential to protect "
    "individual rights and promote fairness. Concerns include bias, privacy "
    "breaches, job displacement and the environmental footprint of AI "
    "infrastructure. Respondents advocated public education programs, AI "
    "literacy initiatives, and community engagement through libraries and "
    "forums. Many expressed strong skepticism toward generative AI, demanding "
    "strict regulation, penalties for non-compliance and frameworks that "
    "uphold Canadian values."
)

# ─── Topics ──────────────────────────────────────────────────────────────────

TOPICS = {
    "education": {
        "csv": DATA_DIR / "ai-strategy-raw-data-2025.csv",
        "summary": EDUCATION_SUMMARY,
        "label": "Education & Skills",
    },
    "trust": {
        "csv": DATA_DIR / "data2.csv",
        "summary": TRUST_SUMMARY,
        "label": "Safe AI & Public Trust",
    },
}

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

TOPIC_RELEVANCE_ANCHORS = {
    "education": [
        # Verbatim CSV consultation questions
        "What skills are required for a modern, digital economy, and how can Canada best support their development and deployment in the workforce?",
        "How can we enhance AI literacy in Canada, including awareness of AI’s limitations and biases?",
        "What can Canada do to ensure equitable access to AI literacy across regions, demographics and socioeconomic groups?",
        # Broad domain coverage — captures fringe/critical stances
        "artificial intelligence education skills workforce Canada",
        "AI digital literacy technology training employment",
        "concerns about AI impact on jobs, education, and society",
    ],
    "trust": [
        # Verbatim CSV consultation questions
        "How can Canada build public trust in AI technologies while addressing the risks they present? What are the most important things to do to build confidence?",
        "What frameworks, standards, regulations and norms are needed to ensure AI products in Canada are trustworthy and responsibly deployed?",
        "How can Canada proactively engage citizens and businesses to promote responsible AI use and trust in its governance? Who is best placed to lead which efforts that fuel trust?",
        # Broad domain coverage
        "artificial intelligence safety accountability transparency regulation Canada",
        "AI ethics privacy bias governance public trust",
        "concerns about AI surveillance, bias, and societal harms",
    ],
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
