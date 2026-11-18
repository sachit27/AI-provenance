from pathlib import Path
import hashlib,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read_records(dataset):
    rows=[json.loads(s) for s in (Path(dataset)/'records.jsonl').read_text().split('\n') if s.strip()]
    if not rows or len({r['respondent_id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate respondent IDs')
    if any(not r['text'].strip() for r in rows):raise ValueError('Empty retained text')
    return rows

def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid embedding matrix')
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(norms<1e-12):raise ValueError('Zero embedding vector')
    return (x/norms).astype(np.float32)
def split_sentences(text):
    # Same transparent splitter as Canadian summary and candidate procedure.
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])',text.strip()) if p.strip()]
def canonical(t):return ' '.join(re.findall(r'\w+',t.lower(),flags=re.UNICODE))
def load_embeddings(dataset,backend):
    d=Path(dataset);rows=read_records(d);sentences=read_json(d/'summary.json')['sentences'];p=d/'embeddings'/f'{backend}.npz';meta=read_json(p.with_suffix('.json'))
    if meta.get('text_digest')!=digest([r['text'] for r in rows]+sentences):raise ValueError(f'{p}: texts/order differ from embedded source')
    saved=np.load(p,allow_pickle=False);x=normalize(saved['records']);y=normalize(saved['summary'])
    if len(x)!=len(rows) or len(y)!=len(sentences) or x.shape[1]!=y.shape[1]:raise ValueError('Embedding block shapes differ from dataset')
    return x,y,meta
