#!/usr/bin/env python3
"""Parse closed (radio / checkbox / matrix) answers from Scotland Citizen Space response pages.

Stdlib only.  Run from the canadian_data root:
    analyses/.venv/bin/python analyses/stance_groups/parse_closed.py

Outputs (analyses/stance_groups/):
    closed_answers.csv  long format: respondent_id, q_num, q_text, sub_item, option, ticked_n
                        one row per TICKED option; a row with empty `option` when the
                        question/sub-item is displayed but nothing is ticked.
                        Questions absent from the page have no rows (= missing).
    parse_qa.json       counts per question/sub-item/option, page counts, unmatched IDs.
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # canadian_data
RAW = ROOT / "analyses/dataset_audit/scotland/raw"
RECS = ROOT / "analyses/scotland/datasets"
OUT = Path(__file__).resolve().parent
WS = re.compile(r"\s+")


class Tok(HTMLParser):
    """Linear token stream: ('h4',text) ('rowhead',text) ('text',text)."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.toks = []
        self.in_h4 = False; self.h4 = []
        self.in_row = False; self.row = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        elif tag == "h4":
            self.in_h4 = True; self.h4 = []
        elif tag == "th" and "cs-matrix-subquestion-row-header" in (a.get("class") or ""):
            self.in_row = True; self.row = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag == "h4" and self.in_h4:
            self.in_h4 = False
            self.toks.append(("h4", WS.sub(" ", " ".join(self.h4)).strip()))
        elif tag == "th" and self.in_row:
            self.in_row = False
            parts = [p for p in self.row if p and p != "Please select one item"]
            self.toks.append(("rowhead", parts[0] if parts else ""))

    def handle_data(self, data):
        if self.skip:
            return
        t = WS.sub(" ", data).strip()
        if not t:
            return
        if self.in_h4:
            self.h4.append(t)
        elif self.in_row:
            self.row.append(t)
        else:
            self.toks.append(("text", t))


STATUS = {"ticked": 1, "unticked": 0, "checked": 1, "not checked": 0}
KIND = re.compile(r"^(Radio button|Checkbox):\s*(.*)$", re.I)
QNUM = re.compile(r"^(\d+)\.\s*(.*)$")


def parse_page(html):
    p = Tok(); p.feed(html)
    toks = p.toks
    answers = {}   # (qnum, sub) -> list of ticked options ; value [] = shown, none ticked
    qtext = {}
    allopts = defaultdict(list)
    q = None; row = ""
    i = 0; n = len(toks)
    while i < n:
        kind, t = toks[i]
        if kind == "h4":
            m = QNUM.match(t)
            if m:
                q = int(m.group(1)); qtext[q] = m.group(2).strip(); row = ""
            i += 1; continue
        if kind == "rowhead":
            row = t; i += 1; continue
        m = KIND.match(t) if kind == "text" else None
        if m and q is not None:
            rest = m.group(2).strip().lower(); j = i + 1
            if not rest:
                # status on following token
                if j < n and toks[j][0] == "text" and toks[j][1].lower() in STATUS:
                    rest = toks[j][1].lower(); j += 1
            if rest in STATUS:
                # label is the next text token
                label = ""
                if j < n and toks[j][0] == "text":
                    label = toks[j][1]; j += 1
                key = (q, row)
                lst = answers.setdefault(key, [])
                if label not in allopts[key]:
                    allopts[key].append(label)
                if STATUS[rest]:
                    lst.append(label)
                i = j; continue
        i += 1
    return qtext, answers, allopts


def main():
    pages = sorted(RAW.glob("response_*.html"))
    rid_re = re.compile(r"response_(\d+)\.html")
    ids = []
    rows = []
    qtexts = {}
    optcount = defaultdict(Counter)       # (q,sub) -> option -> n
    shown = Counter(); answered = Counter()
    optorder = defaultdict(list)
    no_q = []
    for pg in pages:
        rid = rid_re.match(pg.name).group(1)
        ids.append(rid)
        html = pg.read_text(encoding="utf-8", errors="replace")
        qtext, ans, allopts = parse_page(html)
        if not ans:
            no_q.append(rid)
        for q, tx in qtext.items():
            qtexts.setdefault(q, tx)
        for (q, sub), ticked in ans.items():
            shown[(q, sub)] += 1
            for o in allopts[(q, sub)]:
                if o not in optorder[(q, sub)]:
                    optorder[(q, sub)].append(o)
            if ticked:
                answered[(q, sub)] += 1
                for o in ticked:
                    optcount[(q, sub)][o] += 1
                    rows.append((rid, q, qtext[q], sub, o, len(ticked)))
            else:
                rows.append((rid, q, qtext[q], sub, "", 0))
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    with open(OUT / "closed_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["respondent_id", "q_num", "q_text", "sub_item", "option", "ticked_n"])
        w.writerows(rows)

    # match to records.jsonl respondent ids
    rec_ids = {}
    for d in sorted(RECS.iterdir()) if RECS.exists() else []:
        fp = d / "records.jsonl"
        if fp.exists():
            s = set()
            for line in fp.open(encoding="utf-8"):
                if line.strip():
                    s.add(str(json.loads(line)["respondent_id"]))
            rec_ids[d.name] = s
    idset = set(ids)
    allrec = set().union(*rec_ids.values()) if rec_ids else set()
    qa = {
        "n_response_pages": len(pages),
        "n_unique_respondent_ids": len(idset),
        "pages_with_no_closed_questions": no_q,
        "records_unmatched_to_parsed_pages": {k: sorted(v - idset) for k, v in rec_ids.items()},
        "parsed_pages_without_any_free_text_record": len(idset - allrec),
        "free_text_record_counts": {k: len(v) for k, v in rec_ids.items()},
        "questions": {},
    }
    for (q, sub) in sorted(shown, key=lambda k: (k[0], k[1])):
        e = qa["questions"].setdefault(str(q), {"text": qtexts[q], "items": {}})
        e["items"][sub or "(question)"] = {
            "pages_shown": shown[(q, sub)],
            "pages_with_tick": answered[(q, sub)],
            "options_seen": optorder[(q, sub)],
            "option_counts": {o: optcount[(q, sub)].get(o, 0) for o in optorder[(q, sub)]},
        }
    (OUT / "parse_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"pages={len(pages)} rows={len(rows)} questions={len(qa['questions'])}")
    for k, v in qa["records_unmatched_to_parsed_pages"].items():
        print(f"  unmatched {k}: {len(v)} / {qa['free_text_record_counts'][k]}")


if __name__ == "__main__":
    main()
