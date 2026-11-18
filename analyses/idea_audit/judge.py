"""Step 3: three-signal judging of each idea against each summary (cosine, NLI, LLM) + a-priori decision rule."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

import common as C
from stats import cohen_kappa, decide, hypothesis, COVER_LABELS

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

LLM_SYSTEM = """You audit consultation summaries. You are given a SUMMARY (numbered sentences) and one IDEA that some respondents raised, with its stance.
Decide whether the summary expresses that idea. Judge only what the summary says, not whether the idea is correct.
Labels:
- expressed: the summary clearly states this position (wording may differ) with enough specificity that a reader would learn respondents raised it.
- partially: the summary touches the theme or a broader/narrower version, but not the specific position.
- not_expressed: the summary does not mention this idea.
- contradicted: the summary states the opposite position, or characterises respondents as holding the opposite stance.
Return JSON: {"label":"expressed|partially|not_expressed|contradicted","evidence":"<number(s) of the most relevant summary sentence(s), or none>"}"""


class NLI:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(NLI_MODEL, max_length=512)
        self.tok, self.model = ce.tokenizer, ce.model.eval()
        id2l = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = [next(i for i, v in id2l.items() if key in v) for key in ("contradiction", "neutral", "entail")]
        print(f"NLI label order from config: {id2l} -> using [contra, neutral, entail] = {self.idx}")
        self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)
        self.torch = torch

    def probs(self, pairs, bs=16):
        self.load()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), bs):
                ch = pairs[i:i + bs]
                enc = self.tok([p for p, _ in ch], [h for _, h in ch], padding=True, truncation="only_first",
                               max_length=512, return_tensors="pt").to(self.dev)
                pr = self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
                out.extend(pr[:, self.idx].tolist())
        return out


class Judge:
    def __init__(self):
        self.nli = NLI()

    def nli_scores(self, hyp, premises):
        keys = ["nli|" + C.sha(NLI_MODEL, p, hyp) for p in premises]
        have = C.cache().get_many(set(keys))
        miss = sorted({k: p for k, p in zip(keys, premises) if k not in have}.items())
        if miss:
            pr = self.nli.probs([(p, hyp) for _, p in miss])
            C.cache().set_many([(k, v) for (k, _), v in zip(miss, pr)])
            have.update({k: v for (k, _), v in zip(miss, pr)})
        return np.array([have[k] for k in keys])        # [n, 3] = contra, neutral, entail

    def llm(self, statement, stance, sentences):
        summ = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
        out = C.chat_json(LLM_SYSTEM, f"SUMMARY:\n{summ}\n\nIDEA (stance: {stance}): {statement}", max_tokens=120)
        lab = str(out.get("label", "")).lower().strip()
        return (lab if lab in ("expressed", "partially", "not_expressed", "contradicted") else "not_expressed"), str(out.get("evidence", ""))

    def evaluate(self, statement, stance, sentences, use_full=True):
        E = C.embed_texts([statement] + list(sentences))
        sims = E[1:] @ E[0]
        hyp = hypothesis(statement)
        prem = list(sentences)
        full = use_full and len(sentences) > 1
        if full:
            prem.append(" ".join(sentences))
        P = self.nli_scores(hyp, prem)
        ent, con = P[:, 2], P[:, 0]
        lab, ev = self.llm(statement, stance, sentences)
        r = {"cos_max": float(sims.max()), "cos_idx": int(sims.argmax()),
             "nli_entail": float(ent.max()), "nli_contra": float(con.max()),
             "nli_best_idx": int(ent.argmax()) if not (full and ent.argmax() == len(sentences)) else -1,
             "nli_entail_full": float(ent[-1]) if full else float("nan"),
             "nli_sent_entail": [float(x) for x in ent[:len(sentences)]],
             "llm": lab, "llm_evidence": ev}
        r["label"] = decide(r["nli_entail"], r["nli_contra"], lab)
        return r


def prefetch_llm(jobs, workers):
    """jobs: list of (statement, stance, sentences). Warms the cache in parallel."""
    J = Judge()
    C.pmap(lambda j: J.llm(*j), jobs, workers, "llm-judge")


def run(topic, thr, summaries, workers):
    C.set_step("judge")
    ideas = pd.read_parquet(C.topic_dir(topic) / f"ideas_{C.tag_for(thr)}.parquet")
    sums = C.load_summaries(topic, summaries)
    wider = C.wider_sentences(topic)
    print(f"[{topic}] {len(ideas)} ideas x {len(sums)} summaries")
    prefetch_llm([(r.canonical, r.stance, s) for s in sums.values() for r in ideas.itertuples()], workers)
    J = Judge()
    rows = []
    for name, sents in sums.items():
        for r in ideas.itertuples():
            d = J.evaluate(r.canonical, r.stance, sents)
            d.pop("nli_sent_entail")
            rows.append({"idea_id": r.idea_id, "summary": name, **d})
        print(f"  judged vs {name}")
    df = pd.DataFrame(rows)
    # Scotland relocation check: ideas MISSED in the official section target, re-judged on the wider findings text
    if wider and "official" in sums:
        miss = df[(df.summary == "official") & (df.label == "MISSED")].idea_id
        sub = ideas[ideas.idea_id.isin(set(miss))]
        print(f"[{topic}] relocation check: {len(sub)} MISSED ideas vs {len(wider)} wider-findings sentences")
        prefetch_llm([(r.canonical, r.stance, wider) for r in sub.itertuples()], workers)
        extra = []
        for r in sub.itertuples():
            d = J.evaluate(r.canonical, r.stance, wider, use_full=False)
            d.pop("nli_sent_entail")
            extra.append({"idea_id": r.idea_id, "summary": "official_wider", **d})
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    p = C.topic_dir(topic) / f"judgments_{C.tag_for(thr)}.parquet"
    if p.exists():   # keep rows of summaries not re-run now (official now, generated later)
        old = pd.read_parquet(p)
        df = pd.concat([old[~old.summary.isin(set(df.summary))], df], ignore_index=True)
    df.to_parquet(p, index=False)
    nb = (df.nli_entail >= 0.5).astype(int).tolist()
    lb = df.llm.isin(COVER_LABELS).astype(int).tolist()
    print(f"[{topic}] wrote {p}; Cohen's kappa(NLI covered vs LLM covered) = {cohen_kappa(nb, lb):.3f}")
    print(df[df.summary != "official_wider"].groupby("summary").label.value_counts().unstack(fill_value=0).to_string())


def main():
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--summaries", choices=["official", "all"], default="official")
    a = ap.parse_args()
    for t in a.topics:
        run(t, a.threshold, a.summaries, a.workers)
    C.print_costs()


if __name__ == "__main__":
    main()
