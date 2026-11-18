#!/usr/bin/env python3
"""Resumable, content-addressed full-text embeddings; no silent input truncation."""
from pathlib import Path
import argparse,json,sqlite3,sys,urllib.request,time,os,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.common import ROOT,BASE,read_json,read_records,write_json,digest,normalize

def request_json(url,payload=None,timeout=180):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
 for i in range(4):
  try:
   with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
  except Exception:
   if i==3:raise
   time.sleep(2**i)

def token_chunks(text,encode,limit):
 """Lossless character boundaries, checked with actual tokenizer; no decode loss."""
 result=[];remaining=text
 while remaining:
  tokens=encode(remaining)
  if len(tokens)<=limit:result.append((remaining,len(tokens)));break
  low,high=1,len(remaining)
  while low<high:
   mid=(low+high+1)//2
   if len(encode(remaining[:mid]))<=limit:low=mid
   else:high=mid-1
  cut=low
  # Prefer a nearby word boundary, retain whitespace in exactly one chunk.
  space=remaining.rfind(' ',int(.8*cut),cut)
  if space>0:cut=space+1
  part=remaining[:cut];n=len(encode(part))
  if not n or n>limit:raise ValueError('Tokenizer chunk bound could not be satisfied')
  result.append((part,n));remaining=remaining[cut:]
 assert ''.join(c[0] for c in result)==text
 return result

class Embedder:
 def __init__(self,backend,cache,ollama_url=None,model=None):
  self.backend=backend;self.url=(ollama_url or os.environ.get('OLLAMA_URL','http://localhost:11434')).rstrip('/')
  self.model=model or {'openai':'text-embedding-3-large','mpnet':'sentence-transformers/all-mpnet-base-v2','ollama':'qwen3-embedding:8b','nomic':'nomic-embed-text:latest'}[backend]
  self.stats={'texts':0,'chunks':0,'cache_hits':0};self.encoder=None;self.client=None
  if backend=='openai':
   import tiktoken
   from openai import OpenAI
   from dotenv import load_dotenv
   load_dotenv(BASE/'.env');self.client=OpenAI();self.tokenizer=tiktoken.encoding_for_model(self.model);self.limit=8000
   self.identity={'backend':backend,'model':self.model,'chunk_tokens':self.limit,'policy':'full_text_token_weighted_normalized_chunk_mean_v1'}
  elif backend=='mpnet':
   from sentence_transformers import SentenceTransformer
   self.encoder=SentenceTransformer(self.model);self.tokenizer=self.encoder.tokenizer;self.limit=self.encoder.max_seq_length-2
   self.identity={'backend':backend,'model':self.model,'model_commit':getattr(self.encoder[0].auto_model.config,'_commit_hash',None),'chunk_tokens':self.limit,'policy':'full_text_token_weighted_normalized_chunk_mean_v1'}
  else:
   tags=request_json(self.url+'/api/tags')['models'];found=next((x for x in tags if x['name']==self.model or x.get('model')==self.model),None)
   if not found:raise ValueError(f'Exact embedding model tag {self.model!r} not found on {self.url}')
   show=request_json(self.url+'/api/show',{'model':self.model})
   contexts=[int(v) for k,v in show.get('model_info',{}).items() if k.endswith('.context_length')]
   self.limit=min(8192,min(contexts) if contexts else 8192)
   # UTF-8 bytes are a conservative bound for tokenizer input, chunk before API.
   self.identity={'backend':backend,'model':self.model,'digest':found['digest'],'show_digest':digest(show),'chunk_utf8_bytes':self.limit-512,'policy':'full_text_byte_weighted_normalized_chunk_mean_v1'}
  cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(cache);self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB, n INTEGER)')
 def chunk(self,text):
  if self.backend=='openai':
   return token_chunks(text,self.tokenizer.encode,self.limit)
  if self.backend=='mpnet':
   return token_chunks(text,lambda t:self.tokenizer.encode(t,add_special_tokens=False),self.limit)
  chunks=[];part=[];n=0
  for c in text:
   b=len(c.encode())
   if n+b>self.limit-512:chunks.append((''.join(part),n));part=[];n=0
   part.append(c);n+=b
  if part:chunks.append((''.join(part),n))
  return chunks
 def embed(self,texts):
  output=[]
  for lo in range(0,len(texts),32):
   batch=texts[lo:lo+32];pending=[];values=[None]*len(batch)
   for j,text in enumerate(batch):
    if not text.strip():raise ValueError('Cannot embed blank text')
    key=digest([self.identity,text]);cached=self.db.execute('SELECT value,n FROM vectors WHERE key=?',(key,)).fetchone();self.stats['texts']+=1
    if cached:
     values[j]=np.frombuffer(cached[0],dtype=np.float32);self.stats['cache_hits']+=1;self.stats['chunks']+=cached[1]
    else:pending.append((j,key,self.chunk(text)))
   flat=[t for _,_,cs in pending for t,_ in cs];vectors=[]
   for offset in range(0,len(flat),32 if self.backend not in ('ollama','nomic') else 8):
    chunk_texts=flat[offset:offset+(32 if self.backend not in ('ollama','nomic') else 8)]
    if self.backend=='openai':
     result=self.client.embeddings.create(model=self.model,input=chunk_texts);v=[x.embedding for x in sorted(result.data,key=lambda x:x.index)]
    elif self.backend=='mpnet':v=self.encoder.encode(chunk_texts,batch_size=32,normalize_embeddings=True,show_progress_bar=False)
    else:
     result=request_json(self.url+'/api/embed',{'model':self.model,'input':chunk_texts,'truncate':False,'options':{'num_ctx':self.limit}},timeout=600);v=result['embeddings']
    if len(v)!=len(chunk_texts):raise ValueError('Missing embedding chunks')
    vectors.extend(v)
   offset=0
   for j,key,chunks in pending:
    v=normalize(vectors[offset:offset+len(chunks)]);offset+=len(chunks)
    vec=normalize(np.average(v,weights=[c[1] for c in chunks],axis=0)[None,:])[0];values[j]=vec
    self.db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?)',(key,vec.tobytes(),len(chunks)));self.stats['chunks']+=len(chunks)
   self.db.commit();output.extend(values)
   if (lo//32)%10==0:print(f'{self.backend}: {min(lo+32,len(texts))}/{len(texts)} texts',flush=True)
  return normalize(output)

def embed_dataset(dataset,backend,ollama_url=None,model=None):
 d=Path(dataset);rows=read_records(d);ss=read_json(d/'summary.json')['sentences'];texts=[r['text'] for r in rows]+ss
 e=Embedder(backend,ROOT/'cache'/'embeddings.sqlite',ollama_url,model);fingerprint=digest([e.identity,texts]);p=d/'embeddings'/f'{backend}.npz';meta=p.with_suffix('.json')
 if p.exists():
  if not meta.exists() or read_json(meta).get('fingerprint')!=fingerprint:raise ValueError(f'{p}: model/text changed; use a separate backend/output artifact')
  print('Verified cached dataset',d,backend)
 else:
  values=e.embed(texts);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,records=values[:len(rows)],summary=values[len(rows):]);write_json(meta,{'fingerprint':fingerprint,'text_digest':digest(texts),'identity':e.identity,'statistics':e.stats,'n_records':len(rows),'n_summary':len(ss)})
 # Relocation and aggregate-summary sensitivity use exactly the same model identity.
 if (d/'sensitivity_targets.json').exists():
  for name,sents in read_json(d/'sensitivity_targets.json').items():
   target=d/'targets'/backend/f'{name}.npz';target.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(target,summary=e.embed(sents));write_json(target.with_suffix('.json'),{'sentences':sents,'identity':e.identity,'interpretation':'alternative compression/scope target; not budget-matched'})
 return p

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--backend',choices=['openai','mpnet','nomic','ollama'],required=True);p.add_argument('--ollama-url');p.add_argument('--model');a=p.parse_args();embed_dataset(a.dataset,a.backend,a.ollama_url,a.model)
if __name__=='__main__':main()
