#!/usr/bin/env python3
"""Supplementary, source-linked diagnostics. Never modifies primary study artifacts."""
import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.common import ROOT, digest, read_json, read_records, write_json, normalize
from shared.embed import Embedder

DATASETS = ['canada/education', 'canada/trust'] + [f'scotland/q{x}' for x in (4, 7, 20, 21, 22, 23)]
BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c])')
CONDITION = re.compile(r'\b(?:provided that|on condition that|only if|as long as)\b', re.I)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def csvout(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def passages(records):
    """Exact offsets into retained response text, not raw webpage offsets."""
    units, excluded = [], []
    for r in records:
        text = r['text']; starts = [0]; ends = []
        for m in BOUNDARY.finditer(text):
            ends.append(m.start()); starts.append(m.end())
        ends.append(len(text))
        for i, (a, b) in enumerate(zip(starts, ends)):
            a += len(text[a:b]) - len(text[a:b].lstrip())
            b = a + len(text[a:b].rstrip())
            s = text[a:b]
            row = dict(unit_id=f"{r['respondent_id']}:{i}", respondent_id=r['respondent_id'],
                       start=a, end=b, text=s)
            assert text[a:b] == s
            if len(s.split()) < 5:
                excluded.append(dict(row, reason='fewer_than_5_whitespace_words'))
            else:
                units.append(row)
    return units, excluded


def pick(rows, n, seed, kind):
    return sorted(rows, key=lambda u: digest([seed, kind, u['unit_id']]))[:n]


def controls(units, n, seed):
    """Construct controls without using official scores or summaries."""
    eligible = [u for u in units if 8 <= len(u['text'].split()) <= 80]
    # One stable eligible passage per respondent limits verbosity weighting.
    first = {}
    for u in sorted(eligible, key=lambda u: digest([seed, u['unit_id']])):
        first.setdefault(u['respondent_id'], u)
    eligible = list(first.values()); out = []
    for u in pick(eligible, n, seed, 'base'):
        others = [v for v in eligible if v['respondent_id'] != u['respondent_id'] and v['text'] != u['text']]
        if not others:
            continue
        v = min(others, key=lambda v: digest([seed, u['unit_id'], v['unit_id']]))
        a = u['text']; b = v['text']
        common = dict(unit_id=u['unit_id'], respondent_id=u['respondent_id'], source_text=a,
                      distractor_unit_id=v['unit_id'], distractor_text=b)
        out.append(dict(common, kind='exact_passage_deletion', probe=a,
                        before=[a, b], after=[b], control=[a]))
        # Synthetic positions: never attributed to the actual respondent.
        support = 'Position A endorses this statement: ' + a
        oppose = 'Position A rejects this statement: ' + a
        unrelated = 'Position A endorses this statement: ' + b
        out.append(dict(common, kind='synthetic_stance_reversal', probe=support,
                        before=[support], after=[oppose], control=[unrelated]))
        out.append(dict(common, kind='synthetic_disagreement_deletion', probe=oppose,
                        before=[support, oppose], after=[support], control=[oppose]))
    condition_eligible = []
    for u in units:
        matches = list(CONDITION.finditer(u['text']))
        if len(matches) != 1:
            continue
        m = matches[0]
        if len(u['text'][:m.start()].split()) >= 5 and len(u['text'][m.start():].split()) >= 3 and len(u['text'].split()) <= 120:
            condition_eligible.append(u)
    first = {}
    for u in sorted(condition_eligible, key=lambda u: digest([seed, u['unit_id']])):
        first.setdefault(u['respondent_id'], u)
    for u in pick(list(first.values()), n, seed, 'condition'):
        a = u['text']; m = CONDITION.search(a)
        prefix = a[:m.start()].rstrip(); suffix = a[m.start():]
        # Exact deletion of a suffix beginning with an explicit condition marker.
        # Control deletes the same number of whitespace-delimited words at the start.
        words = list(re.finditer(r'\S+', a)); count = len(suffix.split())
        if count >= len(words):
            continue
        matched = a[words[count].start():]
        out.append(dict(kind='condition_marker_suffix_deletion', unit_id=u['unit_id'],
                        respondent_id=u['respondent_id'], source_text=a,
                        removed_text=a[len(prefix):], marker=m.group(),
                        probe=a, before=[a], after=[prefix], control=[matched],
                        deleted_word_count=count))
    return out


def nearest(x, y, k=3):
    """Bounded matrix batches; stable tie breaking follows target order."""
    indices, scores = [], []
    for lo in range(0, len(x), 256):
        sim = np.clip(x[lo:lo+256] @ y.T, -1, 1)
        ix = np.argsort(-sim, axis=1, kind='stable')[:, :min(k, len(y))]
        indices.extend(ix.tolist()); scores.extend(np.take_along_axis(sim, ix, axis=1).tolist())
    return indices, scores


def saved_vectors(path, texts, embedder):
    fp = digest([embedder.identity, texts]); meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        if read_json(meta)['fingerprint'] != fp:
            raise ValueError(f'Changed embedding inputs/identity at {path}; use a new output directory')
        x = np.load(path, allow_pickle=False)['vectors']
        if x.shape[0] != len(texts):
            raise ValueError('Cached vector count differs')
        return normalize(x)
    x = embedder.embed(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp, vectors=x); tmp.replace(path)
    write_json(meta, dict(fingerprint=fp, identity=embedder.identity, n=len(texts)))
    return x


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9))) if len(a) else dict(n=0)


def analyze(dataset, units, records, target, testcases, vectors, backend, out):
    x = normalize([vectors[u['text']] for u in units])
    section = target['section']; wide = target.get('full_findings')
    ix, sc = nearest(x, normalize([vectors[s] for s in section]))
    wi, ws = nearest(x, normalize([vectors[s] for s in wide])) if wide else (None, None)
    scores = np.array([s[0] for s in sc]); ranks = rankdata(scores, method='average') / len(scores)
    record_map = {r['respondent_id']: r['text'] for r in records}
    rows = []; response_scores = defaultdict(list)
    for i, u in enumerate(units):
        matches = [dict(index=j, text=section[j], cosine=v) for j, v in zip(ix[i], sc[i])]
        row = dict(u, backend=backend, section_max_cosine=float(scores[i]),
                   section_percentile=float(ranks[i]), closest_section_passages=matches,
                   source_context=record_map[u['respondent_id']][max(0,u['start']-250):u['end']+250])
        if wide:
            row['closest_full_findings_passages'] = [dict(index=j, text=wide[j], cosine=v) for j,v in zip(wi[i],ws[i])]
            row['full_findings_max_cosine'] = ws[i][0]
            row['scope_gain'] = ws[i][0] - scores[i]
        rows.append(row); response_scores[u['respondent_id']].append(float(scores[i]))
    jsonl(out/'passage_matches.jsonl', rows)
    csvout(out/'respondent_scores.csv', [dict(respondent_id=k, n_passages=len(v), mean_passage_cosine=float(np.mean(v)), minimum_passage_cosine=min(v)) for k,v in response_scores.items()])
    results = []
    for t in testcases:
        probe = vectors[t['probe']]
        score = lambda texts: float(np.clip(max(np.dot(probe, vectors[s]) for s in texts), -1, 1))
        before, after, control = [score(t[k]) for k in ('before','after','control')]
        results.append(dict(t, before_score=before, after_score=after, control_score=control,
                            score_drop=before-after, control_drop=before-control,
                            drop_minus_control_drop=control-after))
    jsonl(out/'controlled_results.jsonl', results)
    grouped = {}
    for kind in sorted({t['kind'] for t in results}):
        selected = [r for r in results if r['kind']==kind]
        grouped[kind] = dict(score_drop=describe([r['score_drop'] for r in selected]),
                             control_drop=describe([r['control_drop'] for r in selected]),
                             drop_minus_control_drop=describe([r['drop_minus_control_drop'] for r in selected]))
    report = dict(dataset=dataset, backend=backend, respondents_with_eligible_passages=len(response_scores),
                  passages=len(units), section_sentence_count=len(section),
                  wider_findings_available=bool(wide),
                  response_equal_weight_mean=float(np.mean([np.mean(v) for v in response_scores.values()])),
                  controls=grouped, inference='Descriptive diagnostic, not validated omission/representation labels; no significance claims.')
    write_json(out/'summary.json', report)
    return rows


def collect(out, datasets, backends):
    overall = []
    for ds in datasets:
        folder = out / ds
        tables = {}
        for backend in backends:
            path = folder/backend/'passage_matches.jsonl'
            if not path.exists():
                continue
            tables[backend] = {r['unit_id']:r for r in (json.loads(s) for s in path.read_text().split('\n') if s.strip())}
        if len(tables) != len(backends):
            continue
        ids = list(next(iter(tables.values())))
        if any(set(t) != set(ids) for t in tables.values()):
            raise ValueError('Backend passage inventories differ')
        rows = []
        for uid in ids:
            rs = [tables[b][uid] for b in backends]; r=rs[0]
            row = dict(unit_id=uid, respondent_id=r['respondent_id'], text=r['text'],
                       worst_backend_percentile=max(t['section_percentile'] for t in rs),
                       mean_percentile=float(np.mean([t['section_percentile'] for t in rs])))
            for b,t in zip(backends,rs):
                row[b+'_cosine'] = t['section_max_cosine']; row[b+'_percentile'] = t['section_percentile']
                row[b+'_closest_text'] = t['closest_section_passages'][0]['text']
                if 'full_findings_max_cosine' in t:
                    row[b+'_full_findings_cosine'] = t['full_findings_max_cosine']
                    row[b+'_full_findings_text'] = t['closest_full_findings_passages'][0]['text']
            rows.append(row)
        rows.sort(key=lambda r:(r['worst_backend_percentile'],r['mean_percentile'],r['unit_id']))
        csvout(folder/'ranked_passages.csv', rows)
        # Fixed examples: five different respondents at the weak-match end.
        examples=[]; seen=set()
        for r in rows:
            if r['respondent_id'] not in seen:
                examples.append(r); seen.add(r['respondent_id'])
            if len(examples)==5: break
        write_json(folder/'fixed_examples.json', examples)
        sensitivity = {str(p):sum(all(r[b+'_percentile']<=p for b in backends) for r in rows) for p in (.05,.1,.2)}
        overall.append(dict(dataset=ds, n_passages=len(rows), backends=backends,
                            low_rank_intersection_counts=sensitivity,
                            note='Relative weak-match rankings; not omission counts or calibrated thresholds.'))
    write_json(out/'collection.json', overall)
    (out/'RESULTS.md').write_text('# Supplementary provenance diagnostics\n\n' +
        '\n'.join(f"- {r['dataset']}: {r['n_passages']} passages; completed backends: {', '.join(r['backends'])}." for r in overall) +
        '\n\nSee each dataset’s ranked_passages.csv, fixed_examples.json and backend/summary.json. '
        'Complete matched quotations are in backend/passage_matches.jsonl. '
        'Rankings flag weak semantic matches, not proven omissions. Controls are synthetic diagnostics, not official-summary evaluations.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backends', nargs='+', choices=['mpnet','openai','nomic'], default=['mpnet','openai','nomic'])
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--output', type=Path, default=HERE/'results')
    p.add_argument('--ollama-url', default=None)
    p.add_argument('--controls-per-kind', type=int, default=100)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--limit-respondents', type=int, default=0, help='Smoke test only; use a separate output directory')
    p.add_argument('--prepare-only', action='store_true', help='Create inventories and control cases without loading models')
    args=p.parse_args()
    if args.controls_per_kind < 1 or args.limit_respondents < 0: p.error('Invalid count')
    out=args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    code={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'shared/common.py',ROOT/'shared/embed.py')}
    config=dict(datasets=args.datasets, backends=args.backends, controls_per_kind=args.controls_per_kind,
                seed=args.seed, limit_respondents=args.limit_respondents, code=code)
    manifest=out/'protocol.json'
    if manifest.exists() and read_json(manifest)!=config:
        raise ValueError('Protocol/code differs from saved run. Use a fresh --output directory.')
    write_json(manifest, config)
    packages={}
    for name in ('numpy','scipy','sentence-transformers','transformers','torch','openai','tiktoken'):
        try: packages[name]=version(name)
        except PackageNotFoundError: pass
    runtime=dict(python=platform.python_version(), packages=packages)
    write_json(out/f'runtime_{digest(runtime)[:12]}.json', runtime)
    prepared={}
    for ds in args.datasets:
        country,topic=ds.split('/'); src=ROOT/country/'datasets'/topic
        records=read_records(src)
        if args.limit_respondents: records=sorted(records,key=lambda r:digest([args.seed,r['respondent_id']]))[:args.limit_respondents]
        units,excluded=passages(records)
        if not units: raise ValueError(f'No eligible passages in {ds}')
        summary=read_json(src/'summary.json'); target={'section':summary['sentences']}
        if (src/'sensitivity_targets.json').exists():
            wide=read_json(src/'sensitivity_targets.json').get('full_findings')
            if wide: target['full_findings']=wide
        testcases=controls(units,args.controls_per_kind,args.seed)
        folder=out/ds; folder.mkdir(parents=True,exist_ok=True)
        provenance=dict(source_directory=str(src), records_digest=digest(records), target_digest=digest(target),
                        dataset=read_json(src/'dataset.json'), summary_metadata=summary,
                        n_retained_responses=len(records), n_eligible_passages=len(units), n_excluded_short_passages=len(excluded))
        sourcefile=folder/'input_provenance.json'
        if sourcefile.exists() and read_json(sourcefile)!=provenance: raise ValueError('Source changed; use a fresh output directory')
        write_json(sourcefile,provenance); write_json(folder/'targets.json',target)
        jsonl(folder/'source_passages.jsonl',units); jsonl(folder/'excluded_passages.jsonl',excluded)
        jsonl(folder/'control_cases.jsonl',testcases)
        prepared[ds]=(src,records,units,target,testcases)
        print(f'{ds}: {len(records)} responses, {len(units)} passages, {len(testcases)} controls',flush=True)
    if args.prepare_only: return
    for backend in args.backends:
        print(f'Loading {backend}',flush=True)
        e=Embedder(backend,out/'cache.sqlite',args.ollama_url)
        try:
            for ds,(src,records,units,target,testcases) in prepared.items():
                expected=read_json(src/'embeddings'/f'{backend}.json')['identity']
                if expected!=e.identity: raise ValueError(f'{ds}/{backend}: model identity differs from saved study. Do not mix versions.')
                folder=out/ds/backend; folder.mkdir(parents=True,exist_ok=True)
                texts=[u['text'] for u in units]+[s for ss in target.values() for s in ss]
                for t in testcases: texts.extend([t['probe']]+t['before']+t['after']+t['control'])
                texts=list(dict.fromkeys(texts))
                print(f'{ds}/{backend}: {len(texts)} unique texts (cached on resume)',flush=True)
                x=saved_vectors(folder/'vectors.npz',texts,e)
                analyze(ds,units,records,target,testcases,dict(zip(texts,x)),backend,folder)
                print(f'Completed {ds}/{backend}',flush=True)
        finally: e.db.close()
        del e
    collect(out,args.datasets,args.backends)
    print(f'FINISHED: {out / "RESULTS.md"}',flush=True)

if __name__=='__main__': main()
