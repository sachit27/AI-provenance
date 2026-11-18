#!/usr/bin/env python3
"""Verify completed diagnostics without network/model calls; write an assessment."""
from pathlib import Path
import importlib.util
import json
import hashlib
import csv
import numpy as np
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('diagnostic_runner', HERE/'run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT=HERE/'results'

def readlines(p):
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def close(a,b):
    if not np.allclose(a,b,atol=2e-6,rtol=2e-6):
        raise AssertionError(f'Numeric mismatch: {a} vs {b}')

def stats(a):
    a=np.array(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max())) if len(a) else dict(n=0)

def main():
    protocol=m.read_json(OUT/'protocol.json')
    for rel,h in protocol['code'].items():
        assert hashlib.sha256((m.ROOT/rel).read_bytes()).hexdigest()==h, rel
    pooled={}; checks=[]; topic_stats=[]; allhashes={}
    for ds in protocol['datasets']:
        folder=OUT/ds; prov=m.read_json(folder/'input_provenance.json')
        source=Path(prov['source_directory']); records=m.read_records(source)
        assert m.digest(records)==prov['records_digest']
        units=readlines(folder/'source_passages.jsonl'); excluded=readlines(folder/'excluded_passages.jsonl')
        rebuilt,exc=m.passages(records); assert rebuilt==units and exc==excluded
        target=m.read_json(folder/'targets.json'); assert m.digest(target)==prov['target_digest']
        assert target['section']==m.read_json(source/'summary.json')['sentences']
        if 'full_findings' in target: assert target['full_findings']==m.read_json(source/'sensitivity_targets.json')['full_findings']
        cases=readlines(folder/'control_cases.jsonl'); assert cases==m.controls(units,protocol['controls_per_kind'],protocol['seed'])
        texts=[u['text'] for u in units]+[s for ss in target.values() for s in ss]
        for t in cases: texts.extend([t['probe']]+t['before']+t['after']+t['control'])
        texts=list(dict.fromkeys(texts)); tables={}; resp_counts={}
        for backend in protocol['backends']:
            path=folder/backend; meta=m.read_json(path/'vectors.json')
            assert meta['identity']==m.read_json(source/'embeddings'/f'{backend}.json')['identity']
            assert meta['fingerprint']==m.digest([meta['identity'],texts])
            vectors=np.load(path/'vectors.npz',allow_pickle=False)['vectors']
            assert len(vectors)==len(texts) and np.isfinite(vectors).all()
            close(np.linalg.norm(vectors,axis=1),np.ones(len(vectors)))
            bytext={s:i for i,s in enumerate(texts)}
            x=m.normalize(vectors[[bytext[u['text']] for u in units]])
            y=m.normalize(vectors[[bytext[s] for s in target['section']]])
            ix,sc=m.nearest(x,y); ranks=rankdata([s[0] for s in sc],method='average')/len(units)
            wi,ws=(None,None)
            if 'full_findings' in target:
                wi,ws=m.nearest(x,m.normalize(vectors[[bytext[s] for s in target['full_findings']]]))
            rows=readlines(path/'passage_matches.jsonl'); assert len(rows)==len(units)
            per_response={}
            for i,(u,r) in enumerate(zip(units,rows)):
                assert all(r[k]==v for k,v in u.items())
                assert [v['index'] for v in r['closest_section_passages']]==ix[i]
                assert [v['text'] for v in r['closest_section_passages']]==[target['section'][j] for j in ix[i]]
                close([v['cosine'] for v in r['closest_section_passages']],sc[i]); close(r['section_max_cosine'],sc[i][0]); close(r['section_percentile'],ranks[i])
                per_response.setdefault(u['respondent_id'],[]).append(sc[i][0])
                if ws:
                    assert [v['index'] for v in r['closest_full_findings_passages']]==wi[i]
                    assert [v['text'] for v in r['closest_full_findings_passages']]==[target['full_findings'][j] for j in wi[i]]
                    close([v['cosine'] for v in r['closest_full_findings_passages']],ws[i]); close(r['full_findings_max_cosine'],ws[i][0]); close(r['scope_gain'],ws[i][0]-sc[i][0])
            report=m.read_json(path/'summary.json'); close(report['response_equal_weight_mean'],np.mean([np.mean(v) for v in per_response.values()]))
            assert report['respondents_with_eligible_passages']==len(per_response)
            with (path/'respondent_scores.csv').open() as f:
                agg=list(csv.DictReader(f))
            assert len(agg)==len(per_response)
            for row in agg:
                v=per_response[row['respondent_id']]; assert int(row['n_passages'])==len(v)
                close(float(row['mean_passage_cosine']),np.mean(v));close(float(row['minimum_passage_cosine']),min(v))
            cr=readlines(path/'controlled_results.jsonl');assert len(cr)==len(cases)
            country=ds.split('/')[0]
            for t,r in zip(cases,cr):
                assert all(r[k]==v for k,v in t.items())
                probe=vectors[bytext[t['probe']]]
                score=lambda ss:float(np.clip(max(np.dot(probe,vectors[bytext[s]]) for s in ss),-1,1))
                before,after,control=[score(t[k]) for k in ('before','after','control')]
                close([r['before_score'],r['after_score'],r['control_score'],r['score_drop'],r['control_drop'],r['drop_minus_control_drop']],
                      [before,after,control,before-after,before-control,control-after])
                pooled.setdefault((country,backend,t['kind']),[]).append(r)
            for kind in report['controls']:
                subset=[r for r in cr if r['kind']==kind]
                for key in ('score_drop','control_drop','drop_minus_control_drop'):
                    values=[r[key] for r in subset]; saved=report['controls'][kind][key]
                    assert saved['n']==len(values)
                    close([saved['mean'],saved['median'],saved['q10'],saved['q90']], [np.mean(values),np.median(values),np.quantile(values,.1),np.quantile(values,.9)])
            low=[r for r in rows if r['section_percentile']<=.1]
            ts=dict(dataset=ds,backend=backend,n_passages=len(units),respondents=len(per_response),
                    excluded_respondents=len(records)-len(per_response),condition_cases=sum(t['kind']=='condition_marker_suffix_deletion' for t in cases))
            if ws:
                ts['low_decile_wider_score_gain']=stats([r['scope_gain'] for r in low])
                ts['all_passages_wider_score_gain']=stats([r['scope_gain'] for r in rows])
            topic_stats.append(ts); tables[backend]={r['unit_id']:r for r in rows}
            checks.append(dict(dataset=ds,backend=backend,passages_verified=len(rows),controls_verified=len(cr),vectors_verified=len(texts)))
            print(f'Verified {ds}/{backend}: {len(rows)} passages, {len(cr)} controls',flush=True)
            for filename in ('summary.json','passage_matches.jsonl','controlled_results.jsonl','vectors.json'):
                f=path/filename;allhashes[str(f.relative_to(OUT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        # Reproduce the complete cross-backend ranking and fixed example selection.
        expected=[]
        for u in units:
            rs=[tables[b][u['unit_id']] for b in protocol['backends']]
            expected.append((max(r['section_percentile'] for r in rs),float(np.mean([r['section_percentile'] for r in rs])),u['unit_id'],u['respondent_id']))
        expected.sort()
        with (folder/'ranked_passages.csv').open() as f: ranked=list(csv.DictReader(f))
        assert [r['unit_id'] for r in ranked]==[r[2] for r in expected]
        examples=[]; seen=set()
        for r in expected:
            if r[3] not in seen:examples.append(r[2]);seen.add(r[3])
            if len(examples)==5:break
        assert [r['unit_id'] for r in m.read_json(folder/'fixed_examples.json')]==examples
    collection=m.read_json(OUT/'collection.json')
    country_counts={}
    for country in ('canada','scotland'):
        entries=[r for r in collection if r['dataset'].startswith(country+'/')]
        country_counts[country]=dict(passages=sum(r['n_passages'] for r in entries),
            weak_match_intersections={p:sum(r['low_rank_intersection_counts'][p] for r in entries) for p in ('0.05','0.1','0.2')})
    pooled_stats=[]
    for (country,backend,kind),rs in pooled.items():
        pooled_stats.append(dict(country=country,backend=backend,kind=kind,n=len(rs),
            score_drop=stats([r['score_drop'] for r in rs]),after_score=stats([r['after_score'] for r in rs]),
            control_drop=stats([r['control_drop'] for r in rs]),
            drop_minus_control_drop=stats([r['drop_minus_control_drop'] for r in rs])))
    result=dict(status='verified',n_topic_backend_runs=len(checks),checks=checks,country_counts=country_counts,
                pooled_controls=pooled_stats,topic_statistics=topic_stats,output_sha256=allhashes,
                caution='Pooled summaries are descriptive of selected cases. Synthetic stance and disagreement tests are algebraically equivalent under symmetric cosine; do not count them as independent evidence.')
    m.write_json(OUT/'POSTRUN_VERIFICATION.json',result)
    lines=['# Completed diagnostic assessment','',
    'All 24 topic/backend runs were verified against saved vectors and source text. No additional embedding or model calls were needed.','',
    '## Source-linked weak matches','',
    '|Corpus|Eligible passages|Bottom 10% in all three backends|','|---|---:|---:|']
    for c,r in country_counts.items():lines.append(f"|{c}|{r['passages']}|{r['weak_match_intersections']['0.1']}|")
    lines += ['', 'These are relative rankings of passages, not numbers of omitted ideas, distinct participants, or minority opinions. The source splitter creates context-dependent fragments. The fixed lowest-ranked examples include idioms, personal details, and general remarks, so the ranking cannot be equated with substantively important loss.', '',
    '## Controlled sensitivity','', '|Corpus|Backend|Exact-passage deletion: mean score drop|Synthetic stance reversal: mean score drop|Reversed-stance mean similarity|Condition-marker deletion: mean score drop (N)|', '|---|---|---:|---:|---:|---:|']
    index={(r['country'],r['backend'],r['kind']):r for r in pooled_stats}
    for c in ('canada','scotland'):
        for b in protocol['backends']:
            e=index[c,b,'exact_passage_deletion'];s=index[c,b,'synthetic_stance_reversal'];k=index[c,b,'condition_marker_suffix_deletion']
            lines.append(f"|{c}|{b}|{e['score_drop']['mean']:.3f}|{s['score_drop']['mean']:.3f}|{s['after_score']['mean']:.3f}|{k['score_drop']['mean']:.3f} ({k['n']})|")
    lines += ['', 'Exact passage deletion is a basic sanity check: the before target contains the exact probe and has a similarity ceiling of approximately one. It is not a difficult natural omission benchmark.', '',
    'Synthetic opposition retains high cosine despite changing endorsement to rejection. This establishes a controlled example of stance insensitivity, not a stance-classification error rate on real consultations. The topic-replacement control generally changes similarity more strongly; inspect pooled_controls in POSTRUN_VERIFICATION.json for all values.', '',
    'Synthetic disagreement deletion and stance reversal use the same support/rejection pair. Symmetric cosine makes their scores effectively identical. They are two presentations of one sensitivity contrast, not independent validation experiments.', '',
    'Condition-marker deletions lower similarity, but beginning-deletion controls also lower it and generally produce larger losses. Conditions are sparse (23 Canadian and 28 Scottish cases), marker-selected and not semantically annotated. The results do not show that conditions are preferentially protected or reliably recognised.', '',
    '## Use in the revision','',
    'Use the evidence trail to make weak matches inspectable, and the controlled tests to define the metric’s scope. The supported measurement is semantic proximity. These results do not validate automatic identification of omitted viewpoints, minority opinions or faithful representation. A claim that high similarity guarantees preserved stance is inconsistent with the controlled results.', '',
    'Scottish wider-report matching is a relocation/scope diagnostic only. A larger target mechanically offers more chances to match; a gain is not proof of substantive preservation. All exact source passages, matches and fixed examples remain available in the original result files.', '',
    'Primary Canadian study outputs are unchanged. This supplementary passage-level extension uses the revised full-text embedding helpers; it is not an exact replication of legacy preprocessing.']
    (OUT/'ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    print('Saved POSTRUN_VERIFICATION.json and ASSESSMENT.md',flush=True)

if __name__=='__main__': main()
