"""Archive public Scottish NSCP consultation pages; resumable, bounded concurrency."""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urljoin,parse_qs,urlparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from bs4 import BeautifulSoup
import json,time,hashlib,re
ROOT=Path(__file__).resolve().parent
(ROOT/'raw').mkdir(parents=True, exist_ok=True)
BASE='https://consult.gov.scot/healthcare-quality-and-improvement/regulation-of-non-surgical-cosmetic-procedures/'
LIST=BASE+'consultation/published_select_respondent'
def fetch(url,path):
 if path.exists(): return path.read_bytes()
 for attempt in range(4):
  try:
   with urlopen(Request(url,headers={'User-Agent':'Research feasibility audit (public consultation archive)'}),timeout=45) as r: data=r.read()
   path.write_bytes(data);time.sleep(.15);return data
  except Exception:
   if attempt==3: raise
   time.sleep(2**attempt)
links={};url=LIST;page=0
while url:
 data=fetch(url,ROOT/'raw'/f'listing_{page:02}.html');s=BeautifulSoup(data,'html.parser')
 for a in s.find_all('a',href=True):
  if 'view_respondent?' in a['href']:
   u=urljoin(url,a['href']);rid=parse_qs(urlparse(u).query)['uuId'][0];links[rid]=u
 nxt=next((a for a in s.find_all('a',href=True) if a.get_text(' ',strip=True)=='Next page'),None)
 url=urljoin(url,nxt['href']) if nxt else None;page+=1
print('LISTINGS',page,'UNIQUE RESPONSES',len(links),flush=True)
(ROOT/'response_urls.json').write_text(json.dumps(links,indent=2))
errors=[];manifest=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 jobs={pool.submit(fetch,u,ROOT/'raw'/f'response_{rid}.html'):(rid,u) for rid,u in links.items()}
 for i,f in enumerate(as_completed(jobs),1):
  rid,u=jobs[f]
  try:
   data=f.result();manifest.append({'id':rid,'url':u,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  except Exception as e:errors.append({'id':rid,'error':str(e)})
  if i%100==0:print('FETCHED',i,'ERRORS',len(errors),flush=True)
(ROOT/'download_manifest.json').write_text(json.dumps({'listing_pages':page,'responses':sorted(manifest,key=lambda x:x['id']),'errors':errors},indent=2))
print('COMPLETE',len(manifest),'ERRORS',errors,flush=True)
