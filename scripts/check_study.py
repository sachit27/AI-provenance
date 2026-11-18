#!/usr/bin/env python3
"""Check the study archive and public code without loading models."""
import ast
import csv
import hashlib
import json
import math
import re
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results' / 'reference'


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    manifest = ARCHIVE / 'SHA256SUMS'
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ARCHIVE / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Reference checksum differs: {relative}')
        listed.add(relative)
    actual = {str(p.relative_to(ARCHIVE)) for p in ARCHIVE.rglob('*') if p.is_file() and p != manifest}
    assert listed == actual, 'Reference manifest has missing or extra files'
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    assert all(str((ARCHIVE / name).relative_to(ROOT)) in tracked for name in listed), 'Reference files must be tracked by Git'
    for path in (ROOT / 'analyses').rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    generation = runpy.run_path(str(ROOT / 'analyses' / 'llm_summaries' / 'common.py'))
    for directory in generation['PIPELINE_MODULE_DIR'].values():
        assert (directory / 'config.py').is_file(), f'Missing pipeline config: {directory}'
        assert (directory / 'analysis_io.py').is_file(), f'Missing embedding reader: {directory}'
    summaries = [json.loads(p.read_text()) for p in (ARCHIVE / 'llm_summaries' / 'outputs').glob('*/*/*/summary.json')]
    assert len(summaries) == 36
    assert len({(s['topic'], s['model_slug'], s['seed']) for s in summaries}) == 36
    for summary in summaries:
        sentences = [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', summary['text'].strip()) if p.strip()]
        assert len(sentences) == summary['n_sentences']
        assert len(summary['text'].split()) == summary['n_words']
        compliant = len(sentences) == summary['budget_sentences'] and summary['n_words'] <= summary['budget_words']
        assert compliant == summary['budget_compliant']
        assert summary['protocol_version'] == '1.1'
    included = [s for s in summaries if s['topic'] != 'q21']
    assert len(included) == 27 and all(s['budget_compliant'] for s in included)
    q21 = [s for s in summaries if s['topic'] == 'q21']
    assert sum(s['n_sentences'] != s['budget_sentences'] for s in q21) == 7
    scotland = rows(ARCHIVE / 'scotland_analysis' / 'scotland_analysis_summary.csv')
    primary = [r for r in scotland if r['model'] in ('openai', 'mpnet')]
    assert len(primary) == 12
    for row in primary:
        for objective in ('mean_opt', 'tail_opt'):
            gain = float(row[f'{objective}_gain'])
            assert math.isfinite(gain) and gain > 0
            assert float(row[f'{objective}_gain_ci_low']) <= gain <= float(row[f'{objective}_gain_ci_high'])
            assert 0 < float(row[f'{objective}_p_holm']) < 0.05
        assert float(row['exact_random_mean_p']) >= 0.05
    for path in ARCHIVE.rglob('*'):
        assert path.name not in ('records.jsonl', 'closed_answers.csv', 'respondent_groups.csv', 'participant_coverage.csv')
        assert path.suffix not in ('.sqlite', '.npz', '.parquet', '.html')
    print(f'PASS: {len(listed)} study reference files, 36 summaries, 24 benchmark comparisons and Python syntax')


if __name__ == '__main__':
    main()
