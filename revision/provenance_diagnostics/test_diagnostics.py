import tempfile
import unittest
from pathlib import Path
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class DiagnosticsTests(unittest.TestCase):
    def records(self):
        return [dict(respondent_id='a', text='  I endorse national training standards provided that remote clinics retain flexibility.  Another substantial statement describes funding for accessible training. Yes.'),
                dict(respondent_id='b', text='Local implementation needs staff and reliable inspection resources to operate well.')]

    def test_offsets_and_short_exclusions(self):
        records=self.records(); units,excluded=m.passages(records)
        sources={r['respondent_id']:r['text'] for r in records}
        self.assertEqual(len(units),3); self.assertEqual(len(excluded),1)
        for u in units+excluded:
            self.assertEqual(u['text'],sources[u['respondent_id']][u['start']:u['end']])
        self.assertEqual(excluded[0]['text'],'Yes.')

    def test_controls_deterministic_and_condition_exact(self):
        units,_=m.passages(self.records()); cases=m.controls(units,100,1729)
        self.assertEqual(cases,m.controls(list(reversed(units)),100,1729))
        c=next(t for t in cases if t['kind']=='condition_marker_suffix_deletion')
        self.assertEqual(c['after'][0]+c['removed_text'],c['source_text'])
        self.assertEqual(len(c['source_text'].split())-len(c['control'][0].split()),c['deleted_word_count'])
        for t in cases:
            self.assertTrue(all(s.strip() for key in ('before','after','control') for s in t[key]))

    def test_duplicate_response_templates_not_used_as_distractors(self):
        units=[dict(unit_id='a:0',respondent_id='a',text='This is one long enough identical response for testing.'),dict(unit_id='b:0',respondent_id='b',text='This is one long enough identical response for testing.')]
        self.assertEqual(m.controls(units,2,1729),[])

    def test_nearest_stable_ties(self):
        ix,sc=m.nearest(np.array([[1.,0.]]),np.array([[1.,0.],[1.,0.],[0.,1.]]))
        self.assertEqual(ix,[[0,1,2]]); self.assertEqual(sc,[[1.,1.,0.]])

    def test_deleted_duplicate_still_covered(self):
        # Demonstrates that removing a sentence is not guaranteed to remove its information.
        x=np.array([[1.,0.]])
        _,sc=m.nearest(x,np.array([[1.,0.]]))
        self.assertEqual(sc[0][0],1.)

    def test_end_to_end_analytic_vectors(self):
        records=[dict(respondent_id='a',text='One source passage contains\u2028enough words.'),dict(respondent_id='b',text='Another source passage contains enough words.')]
        units,_=m.passages(records); texts=[u['text'] for u in units]
        vec={texts[0]:np.array([1.,0.]),texts[1]:np.array([0.,1.])}
        case=dict(kind='exact_passage_deletion',unit_id='a:0',respondent_id='a',probe=texts[0],before=texts,after=[texts[1]],control=[texts[0]])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); out=root/'canada/education/mpnet'; out.mkdir(parents=True)
            rows=m.analyze('canada/education',units,records,{'section':[texts[0]],'full_findings':texts},[case],vec,'mpnet',out)
            self.assertEqual(rows[1]['section_max_cosine'],0.)
            self.assertEqual(rows[1]['full_findings_max_cosine'],1.)
            result=m.read_json(out/'summary.json')
            self.assertEqual(result['response_equal_weight_mean'],.5)
            self.assertEqual(result['controls']['exact_passage_deletion']['score_drop']['mean'],1.)
            m.collect(root,['canada/education'],['mpnet'])
            self.assertTrue((root/'canada/education/ranked_passages.csv').exists())
            self.assertEqual(m.read_json(root/'collection.json')[0]['n_passages'],2)

    def test_cache_identity_guard(self):
        class Fake:
            identity={'model':'fake'}
            def embed(self,texts): return np.array([[1.,0.] for _ in texts])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'v.npz'
            m.saved_vectors(path,['hello'],Fake())
            with self.assertRaises(ValueError): m.saved_vectors(path,['changed'],Fake())

if __name__=='__main__': unittest.main()
