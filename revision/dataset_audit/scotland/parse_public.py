#!/usr/bin/env python3
"""Extract public Citizen Space response DOM, preserving omitted versus empty fields.
No NLP filtering, deduplication or imputation is applied to the archived records.
"""
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import hashlib, json, re
ROOT=Path(__file__).resolve().parent
TARGETS=(4,7,20,21,22,23)
BASE='https://consult.gov.scot/healthcare-quality-and-improvement/regulation-of-non-surgical-cosmetic-procedures/consultation/view_respondent?uuId='
def norm(t): return re.sub(r'\s+',' ',t).strip()
def empty_answer(t):
    return bool(re.fullmatch(r'(?:n/?a|none|nil|no|no comments?|no further comments?|nothing(?: to add)?|not applicable|don[’\']?t know|do not know|unsure|no opinion|no views|not sure|\-+|\.+)[.!\s]*',norm(t),re.I))
def main():
    rows=[]; errors=[]; questions=defaultdict(Counter); flags=Counter(); metadata=Counter(); summaries={str(q):Counter() for q in TARGETS}; dup=defaultdict(lambda:defaultdict(list)); redactions=[]; attachments=[]
    for p in sorted((ROOT/'raw').glob('response_*.html')):
        rid=p.stem.removeprefix('response_'); raw=p.read_bytes(); html=raw.decode('utf-8'); s=BeautifulSoup(html,'html.parser')
        h=s.find('h2',string=re.compile(r'Response\s+\d+'))
        if h is None or norm(h.get_text())!='Response '+rid:
            errors.append({'file':p.name,'error':'Response heading absent or ID mismatch'}); continue
        blocks={}; repeated=[]
        for title in s.select('h4'):
            m=re.match(r'(\d+)\.\s*(.*)',norm(title.get_text(' ',strip=True)))
            if m:
                q=int(m[1]); repeated += [q] if q in blocks else []; blocks[q]=(title,title.parent)
        rec={'respondent_id':rid,'source_url':BASE+rid,'source_file':'raw/'+p.name,'sha256':hashlib.sha256(raw).hexdigest(),'questions':{},'closed_responses':{},'attachment_links':[],'redaction_mentions':[]}
        if repeated: errors.append({'file':p.name,'error':'Repeated numbered questions','questions':repeated})
        for q,(title,b) in blocks.items():
            # Select visible option only where DOM has explicit check icon; do not use all labels.
            selected=[]
            for opt in b.select('.cs-fake-radio-checkbox-container'):
                if opt.select_one('.fa-check'):
                    label=opt.select_one('.cs-fake-radio-checkbox-label')
                    if label: selected.append(norm(label.get_text(' ',strip=True)))
            if selected: rec['closed_responses'][str(q)]=selected
            for a in b.select('a[href]'):
                href=a['href']
                if re.search(r'attach|download|\.pdf|\.docx?|\.xlsx?',href,re.I):
                    rec['attachment_links'].append({'question':q,'href':href,'label':norm(a.get_text(' ',strip=True))})
        for q in TARGETS:
            key=str(q); count=summaries[key]; count['respondents']+=1
            if q not in blocks:
                rec['questions'][key]={'status':'not_displayed','text':None,'question':None}; count['not_displayed']+=1; continue
            title,b=blocks[q]; qt=norm(title.get_text(' ',strip=True)); questions[key][qt]+=1
            vals=b.select('[data-test-hook="subquestion-textarea"]')
            if not vals:
                errors.append({'file':p.name,'question':q,'error':'Question displayed but no textarea'}); txt=None; status='no_textarea'; count[status]+=1
            else:
                if len(vals)!=1: errors.append({'file':p.name,'question':q,'error':'Multiple textareas','n':len(vals)})
                txt='\n\n'.join(v.get_text('\n',strip=True) for v in vals); status='nonempty' if norm(txt) else 'empty'; count[status]+=1
            no_comment=empty_answer(txt) if txt else False
            rec['questions'][key]={'question':qt,'status':status,'text':txt,'no_comment_exact_rule':no_comment}
            if no_comment: count['no_comment_exact_rule']+=1
            if txt and norm(txt) and not no_comment:
                count['candidate_text_after_exact_no_comment_rule']+=1
                dup[key][norm(txt).casefold()].append(rid)
            if txt and re.search(r'redact|removed|withheld|confidential|personal information|attach',txt,re.I):
                rec['redaction_mentions'].append({'question':q,'text':txt})
        if rec['attachment_links']: attachments.append({'respondent_id':rid,'links':rec['attachment_links']})
        if rec['redaction_mentions']: redactions.append({'respondent_id':rid,'mentions':rec['redaction_mentions']})
        for v in rec['closed_responses'].get('25',[]): metadata[v]+=1
        rows.append(rec)
    with (ROOT/'public_responses.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    duplicates={q:[{'respondent_ids':ids,'text':txt} for txt,ids in groups.items() if len(ids)>1] for q,groups in dup.items()}
    report={'response_files_seen':len(list((ROOT/'raw').glob('response_*.html'))),'parsed_valid_response_pages':len(rows),'unique_respondent_ids':len(set(r['respondent_id'] for r in rows)),'question_counts':summaries,'question_wording_variants':questions,'respondent_type_selected_Q25':metadata,'parsing_errors':errors,'duplicate_text_groups_per_question':{q:len(v) for q,v in duplicates.items()},'duplicate_text_records_per_question':{q:sum(len(g['respondent_ids']) for g in v) for q,v in duplicates.items()},'pages_with_attachment_links':len(attachments),'pages_with_possible_redaction_or_attachment_text':len(redactions),'notes':['not_displayed means absent from published page; it does not establish whether originally blank or withheld.','No-comment rule is conservative and lexical only; candidates are not automatically judged substantive.','Duplicate texts are preserved because unit of analysis is respondent.','Redaction and attachment text flags are broad lexical leads, not verified redactions.','Closed responses only extract explicit checked icons in simple radio/checkbox controls; matrix questions are not extracted.']}
    (ROOT/'parse_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'duplicate_text_qa.json').write_text(json.dumps(duplicates,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'attachment_redaction_qa.json').write_text(json.dumps({'attachments':attachments,'possible_text_mentions':redactions},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
