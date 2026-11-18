"""Pure-python helpers (no third-party imports) so they can be unit-tested anywhere."""
from __future__ import annotations
import math
import re

NLI_ENTAIL_T = 0.5       # a priori
NLI_CONTRA_T = 0.5       # a priori
COVER_LABELS = ("expressed", "partially")


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(a, b):
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def decide(entail: float, contra: float, llm: str) -> str:
    """Two-judge decision rule, v1.1. STANCE_MISMATCH is decided by the stance-aware LLM judge only.
    v1.0 also triggered it on NLI contradiction >= 0.5; the pre-specified negative control
    (unrelated summary) showed MNLI 'contradiction' fires on unrelated text, so NLI
    contradiction is reported but no longer assigns a label."""
    if llm == "contradicted":
        return "STANCE_MISMATCH"
    if entail < NLI_ENTAIL_T and llm == "not_expressed":
        return "MISSED"
    if entail >= NLI_ENTAIL_T and llm in COVER_LABELS:
        return "COVERED"
    return "UNCERTAIN"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', s or ""))).strip().casefold()


def quote_verified(quote: str, text: str) -> bool:
    q = norm_ws(quote)
    return len(q) >= 3 and q in norm_ws(text)


def words(s: str):
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def trigram_overlap(a: str, b: str) -> int:
    wa, wb = words(a), words(b)
    ta = {tuple(wa[i:i + 3]) for i in range(len(wa) - 2)}
    tb = {tuple(wb[i:i + 3]) for i in range(len(wb) - 2)}
    return len(ta & tb)


_PROPER = {"canada", "canadian", "canadians", "scotland", "scottish", "indigenous", "french", "english",
           "government", "ai", "gdpr", "nhs", "uk", "european", "america", "american", "first"}


def lowerfirst(s: str) -> str:
    """Lower-case the first letter unless the first word looks like an acronym / proper noun."""
    s = (s or "").strip().rstrip(".")
    m = re.match(r"[A-Za-z][A-Za-z'-]*", s)
    if not m:
        return s
    w = m.group(0)
    if any(ch.isupper() for ch in w[1:]) or w.lower() in _PROPER and w.lower() not in ("first",):
        return s
    return s[:1].lower() + s[1:]


def hypothesis(statement: str) -> str:
    return "Respondents said that " + lowerfirst(statement)
