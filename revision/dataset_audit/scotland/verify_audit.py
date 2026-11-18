"""Independently reconcile public archive and bs4 extraction using lxml."""
from pathlib import Path
from lxml import html
from collections import Counter
import json,re,hashlib
root=Path(__file__).resolve().parent
norm=lambda s:re.sub(r'\s+',' ',s).strip()
rows=[json.loads(x) for x in (root/'public_responses.jsonl').read_text().split('\n') if x.strip()]
links=json.loads((root/'response_urls.json').read_text());manifest=json.loads((root/'download_manifest.json').read_text())
assert len(rows)==len(links)==len(manifest['responses'])==1809
assert set(links)=={r['respondent_id'] for r in rows}
assert not manifest['errors']
checks={};mismatches=[];redacted=[]
for r in rows:
 p=root/r['source_file'];data=p.read_bytes();tree=html.fromstring(data)
 assert hashlib.sha256(data).hexdigest()==r['sha256']
 qs={}
 for h in tree.xpath('//h4'):
  m=re.match(r'(\d+)\.',norm(h.text_content()))
  if m:qs[m[1]]=h
 for q,v in r['questions'].items():
  if q not in qs:
   if v['status']!='not_displayed':mismatches.append([r['respondent_id'],q,'absent'])
   continue
  vals=qs[q].getparent().xpath('.//*[@data-test-hook="subquestion-textarea"]')
  independent=' '.join(norm(' '.join(x.itertext())) for x in vals)
  if norm(v.get('text') or '')!=norm(independent):mismatches.append([r['respondent_id'],q,'text mismatch'])
  if re.search(r'\bredacted text\b',v.get('text') or '',re.I):redacted.append({'respondent_id':r['respondent_id'],'question':q})
assert not mismatches,mismatches
checks={'archive_records':1809,'listing_pages':manifest['listing_pages'],'failed_downloads':0,'unique_ids':len(links),'all_sha256_match':True,'independent_lxml_text_mismatches':mismatches,'respondents_with_any_nonempty_free_text':sum(any(norm(v.get('text') or '') for v in r['questions'].values()) for r in rows),'nonempty_respondent_question_answers':sum(bool(norm(v.get('text') or '')) for r in rows for v in r['questions'].values()),'explicit_redacted_text_records':redacted,'public_fraction':1809/2207,'nonempty_counts':{q:sum(bool(norm(r['questions'][q].get('text') or '')) for r in rows) for q in ('4','7','20','21','22','23')}}
(root/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
