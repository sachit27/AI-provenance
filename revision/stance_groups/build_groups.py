#!/usr/bin/env python3
"""Derive respondent stance/type groups from closed_answers.csv, following MAPPING.json.
Stdlib only.  Run from canadian_data root:
    revision/.venv/bin/python revision/stance_groups/build_groups.py
Writes respondent_groups.csv (one row per parsed respondent; empty = missing)."""
import csv, re
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
norm = lambda s: s.replace("’", "'").strip().lower()

SO = {"strongly support": "support", "somewhat support": "support", "neutral": "neutral",
      "somewhat oppose": "oppose", "strongly oppose": "oppose"}
AG = {"agree": "support", "strongly agree": "support", "disagree": "oppose",
      "strongly disagree": "oppose", "i don't know": "neutral", "i don't know.": "neutral",
      "neither agree nor disagree": "neutral", "don't know": "neutral"}


def q17(o):
    o = norm(o)
    if o.startswith("there should be a lower age limit"): return "support"
    if o.startswith("there should be no lower age limit"): return "oppose"
    if o.startswith("i don't know"): return "neutral"
    return ""


def grp_num(o):
    o = norm(o)
    if o.startswith("no regulation"): return 0
    m = re.search(r"\(group (\d)\)", o)
    return int(m.group(1)) if m else None  # None = don't know


def main():
    ans = defaultdict(lambda: defaultdict(list))   # rid -> (q,sub) -> [options]
    with open(D / "closed_answers.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ans[r["respondent_id"]][(int(r["q_num"]), r["sub_item"])]
            if r["option"]:
                ans[r["respondent_id"]][(int(r["q_num"]), r["sub_item"])].append(r["option"])
    cols = ["respondent_id", "resp_type", "g_q4_level", "g_q4_direction", "q4_n_answered",
            "q4_share_as_proposed", "g_q7_Q5", "g_q7_Q6", "g_q23_Q19", "g_q23_Q17"]
    out = []
    for rid in sorted(ans):
        a = ans[rid]
        def single(q, mp):
            for (qq, sub), opts in a.items():
                if qq == q and opts:
                    return mp(opts[0]) if callable(mp) else mp.get(norm(opts[0]), "")
            return ""
        typ = single(25, lambda o: norm(o))
        # Q4: compare chosen group with proposed group (Q1->1, Q2->2, Q3->3)
        same = up = down = dk = 0
        shown = False
        for (qq, sub), opts in a.items():
            if qq in (1, 2, 3):
                shown = True
                if not opts: continue
                g = grp_num(opts[0])
                if g is None: dk += 1
                elif g == qq: same += 1
                elif g > qq: up += 1
                else: down += 1
        n = same + up + down
        if not shown or (n == 0 and dk == 0):
            lvl = dirn = ""; share = ""
        elif n == 0:
            lvl, dirn, share = "neutral", "dk", ""
        else:
            share = round(same / n, 4)
            lvl = "support" if same / n >= 0.5 else "oppose"
            dirn = "as_proposed" if lvl == "support" else ("stricter" if up >= down else "looser")
        out.append({"respondent_id": rid, "resp_type": typ, "g_q4_level": lvl, "g_q4_direction": dirn,
                    "q4_n_answered": n, "q4_share_as_proposed": share,
                    "g_q7_Q5": single(5, SO), "g_q7_Q6": single(6, SO),
                    "g_q23_Q19": single(19, AG), "g_q23_Q17": single(17, q17)})
    with open(D / "respondent_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
    from collections import Counter
    for c in cols[1:4] + cols[6:]:
        print(c, dict(Counter(r[c] for r in out)))


if __name__ == "__main__":
    main()
