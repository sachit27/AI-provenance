#!/usr/bin/env python3
"""Write the Canada-layout CSV inputs that pipeline stage 01 expects.

For each Scottish question q: ``inputs/<q>.csv`` with first column ``Internal ID``
(= respondent_id, preserved verbatim) and one free-text column whose header is
the verbatim consultation question.  Stage 01 turns ``Internal ID`` into
``clean.parquet`` columns ``Internal ID`` and ``id`` (= "P" + Internal ID).

Also asserts that the unchanged pipeline sentence splitters re-derive exactly the
official-report sentences stored in summary.json.  No third-party dependency.
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DATASETS = BASE / "revision" / "scotland" / "datasets"
OUT = HERE / "inputs"
QUESTIONS = ["q4", "q7", "q20", "q21", "q22", "q23"]


def split_summary(text):          # 02_prepare_benchmarks.py / analysis_io.py
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text.strip()) if p.strip()]


def split_sentences(text):        # 01_preprocess.py
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text.strip())
            if len(s.strip()) > 10]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for q in QUESTIONS:
        directory = DATASETS / q
        sentences = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["sentences"]
        joined = " ".join(s.strip() for s in sentences)
        assert split_summary(joined) == [s.strip() for s in sentences], f"{q}: summary split mismatch"
        assert split_sentences(joined) == [s.strip() for s in sentences], f"{q}: stage-01 split mismatch"
        question = json.loads((directory / "questions.json").read_text(encoding="utf-8"))["question"]
        records = [json.loads(line) for line in
                   (directory / "records.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        ids = [str(r["respondent_id"]) for r in records]
        assert len(set(ids)) == len(ids), f"{q}: duplicate respondent_id"
        with (OUT / f"{q}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Internal ID", question])
            for record in records:
                writer.writerow([str(record["respondent_id"]), record["text"]])
        words = sum(len(s.split()) for s in sentences)
        print(f"{q}: {len(records)} records; official target {len(sentences)} sentences, {words} words")


if __name__ == "__main__":
    main()
