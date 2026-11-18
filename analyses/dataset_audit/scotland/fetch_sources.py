from pathlib import Path
from urllib.request import urlopen
from bs4 import BeautifulSoup
import hashlib,json
root=Path(__file__).resolve().parent/'sources';root.mkdir(exist_ok=True)
urls={
'consultation':'https://consult.gov.scot/healthcare-quality-and-improvement/regulation-of-non-surgical-cosmetic-procedures/',
'methodology':'https://www.gov.scot/publications/regulation-licensing-non-surgical-cosmetic-procedures-consultation-analysis-response/pages/3/',
'findings':'https://www.gov.scot/publications/regulation-licensing-non-surgical-cosmetic-procedures-consultation-analysis-response/pages/4/',
'background':'https://www.gov.scot/publications/consultation-background-paper-regulation-licensing-non-surgical-cosmetic-procedures/',
'errata':'https://www.gov.scot/publications/regulation-licensing-non-surgical-cosmetic-procedures-consultation-analysis-response/',
'citizen_space_terms':'https://consult.gov.scot/terms_and_conditions/',
'consult_transparency':'https://www.gov.uk/algorithmic-transparency-records/dsit-consult'}
manifest=[]
for name,url in urls.items():
 try:
  with urlopen(url,timeout=40) as r:data=r.read(); final=r.url
  (root/f'{name}.html').write_bytes(data)
  soup=BeautifulSoup(data,'html.parser');body=soup.select_one('.body-content') or soup.select_one('main') or soup
  (root/f'{name}.txt').write_text(body.get_text('\n',strip=True))
  manifest.append({'name':name,'url':url,'final_url':final,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 except Exception as e:manifest.append({'name':name,'url':url,'error':str(e)})
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
