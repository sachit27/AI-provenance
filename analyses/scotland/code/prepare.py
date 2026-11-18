#!/usr/bin/env python3
"""Prepare all six Scottish domains using a frozen, explicit official-paragraph map."""
import sys,json,re,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared.common import ROOT,read_json,write_json,split_sentences,digest
AUDIT=ROOT/'dataset_audit'/'scotland'
FINDINGS='https://www.gov.scot/publications/regulation-licensing-non-surgical-cosmetic-procedures-consultation-analysis-response/pages/4/'
# Verbatim paragraph indices frozen before coverage analysis. All textual findings
# on the question retained; questionnaire explanations/count-only paragraphs omitted.
MAP={
'q4':{'Respondent views on procedures proposed as Group 1':[2,3,4], 'Respondent views on procedures proposed as Group 2':[3,4,5,6,7,8,9,11], 'Respondent views on procedures proposed as Group 3':[2,4,5], 'Wider comments from consultation respondents and stakeholders provided on the proposed groups':[0,1,2,4,5,6]},
'q7':{'Practitioner Licenses (Questions 5-7)':[4,5,6]},
'q20':{'Protected characteristics':[0,1,2]},
'q21':{'How regulation of sector might affect people differently based on their financial situation (Question 21)':[1,2,3,4],'Impact of regulation on businesses':[0,1,2,3]},
'q22':{'Island communities':[0,1]},
'q23':{'Child rights and wellbeing':[0,1,2]}}
LABELS={'q4':'Procedure grouping and regulation','q7':'Licensing','q20':'Protected characteristics','q21':'Financial situation and business impacts','q22':'Island communities','q23':'Children rights'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'scotland'/'datasets');p.add_argument('--min-words',type=int,default=5);a=p.parse_args()
 qa=read_json(AUDIT/'verification.json');assert qa['archive_records']==1809 and not qa['independent_lxml_text_mismatches']
 rows=[json.loads(l) for l in (AUDIT/'public_responses.jsonl').read_text().split('\n') if l.strip()];sections=read_json(AUDIT/'official_findings_sections.json')
 full=[t for k,v in sections.items() if k not in ('Summary of findings','Detail of findings by consultation question asked') for t in v]
 targets={'full_findings':split_sentences(' '.join(full)), 'overall_summary':split_sentences(' '.join(sections['Summary of findings']))}
 for topic,mapping in MAP.items():
  out=a.output/topic;out.mkdir(parents=True,exist_ok=True);q=topic[1:];retained=[];excluded=[]
  for r in rows:
   ans=r['questions'][q];raw=ans.get('text') or '';text=re.sub(r'\bRedacted text\b',' ',raw,flags=re.I);text=re.sub(r'\s+',' ',text).strip()
   reason='not_displayed' if ans['status']=='not_displayed' else 'empty' if not text else 'below_five_words' if len(text.split())<a.min_words else None
   if reason:excluded.append({'respondent_id':r['respondent_id'],'reason':reason});continue
   retained.append({'respondent_id':r['respondent_id'],'text':text,'question_id':q,'source_url':r['source_url'],'redacted':bool(re.search(r'\bRedacted text\b',raw,re.I)),'respondent_type':r['closed_responses'].get('25',[]),'word_count':len(text.split())})
  paragraphs=[{'section':k,'index':i,'text':sections[k][i]} for k,idx in mapping.items() for i in idx]
  sentences=split_sentences(' '.join(x['text'] for x in paragraphs))
  (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in retained))
  write_json(out/'summary.json',{'sentences':sentences,'source_url':FINDINGS,'source_sections':mapping,'paragraphs':paragraphs,'paragraph_source_hash':digest(sections)})
  question=next(r['questions'][q]['question'] for r in rows if r['questions'][q].get('question'))
  write_json(out/'questions.json',{'question_id':q,'question':question,'label':LABELS[topic]})
  write_json(out/'sensitivity_targets.json',targets)
  write_json(out/'exclusions.json',excluded)
  write_json(out/'dataset.json',{'dataset':'Scotland NSCP 2024-25','topic':topic,'label':LABELS[topic],'n':len(retained),'min_words':a.min_words,'source_archive_records':1809,'all_consultation_responses':2207,'public_selection':'consented subset','preprocessing':'strip explicit editorial redaction marker; whitespace normalization; minimum word count; no automatic language/relevance exclusion','scope':'published respondent text vs hybrid official synthesis; section targets plus full-findings relocation sensitivity','record_digest':digest(retained),'summary_digest':digest(sentences),'summary_words':sum(len(s.split()) for s in sentences),'summary_sentences':len(sentences)})
  print(topic,'records',len(retained),'summary sentences',len(sentences),'words',sum(len(s.split()) for s in sentences))
if __name__=='__main__':main()
