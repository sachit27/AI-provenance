#!/usr/bin/env python3
"""Export canonical submitted Canada data without changing the original artifacts."""
from pathlib import Path
import sys,json,argparse,hashlib,importlib.util
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.common import BASE,ROOT,write_json,split_sentences,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'canada'/'datasets');a=p.parse_args()
 sys.path.insert(0,str(BASE/'src'))
 from config import TOPICS,TOPIC_RELEVANCE_ANCHORS
 from analysis_io import embedding_blocks
 for topic in ('education','trust'):
  d=a.output/topic;d.mkdir(parents=True,exist_ok=True);src=BASE/'src'/'analysis'/topic
  df=pd.read_parquet(src/'clean.parquet');rows=[{'respondent_id':str(row['id']),'text':row['text'],'word_count':len(row['text'].split())} for row in df.to_dict('records')]
  (d/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  sentences=split_sentences(TOPICS[topic]['summary']);write_json(d/'summary.json',{'sentences':sentences,'source_url':'https://ised-isde.canada.ca/','source_sections':['submitted official summary'],'provenance':'src/config.py verbatim submitted transcription; AiStrategyReport_EN.pdf page8'})
  write_json(d/'questions.json',{'questions':TOPIC_RELEVANCE_ANCHORS[topic][:3],'label':TOPICS[topic]['label']})
  write_json(d/'dataset.json',{'dataset':'Canada AI Strategy 2025','topic':topic,'n':len(rows),'record_digest':digest(rows),'summary_digest':digest(sentences),'preprocessing':'submitted retained corpus, unchanged','source_clean_sha256':hashlib.sha256((src/'clean.parquet').read_bytes()).hexdigest()})
  for b in ('openai','mpnet'):
   v=embedding_blocks(topic,b);name='legacy_'+b;(d/'embeddings').mkdir(exist_ok=True)
   np.savez_compressed(d/'embeddings'/f'{name}.npz',records=v['records'],summary=v['summary'])
   write_json(d/'embeddings'/f'{name}.json',{'backend':name,'policy':'original submitted embeddings; preserve truncation behavior','source_sha256':hashlib.sha256((src/f'embeddings_{b}.npz').read_bytes()).hexdigest(),'text_digest':digest([r['text'] for r in rows]+sentences)})
  # Original Nomic vectors are a legacy robustness artifact (original 4000-character cap).
  nomic_path=src/'embeddings_nomic_embed_text.npz';nomic=np.load(nomic_path,allow_pickle=False)
  if len(nomic['input_embeddings'])!=len(rows) or len(nomic['summary_embeddings'])!=len(sentences):raise ValueError('Legacy Nomic alignment mismatch')
  np.savez_compressed(d/'embeddings'/'legacy_nomic.npz',records=nomic['input_embeddings'],summary=nomic['summary_embeddings'])
  write_json(d/'embeddings'/'legacy_nomic.json',{'backend':'legacy_nomic','policy':'original submitted Nomic embeddings; original character cap retained','source_sha256':hashlib.sha256(nomic_path.read_bytes()).hexdigest(),'text_digest':digest([r['text'] for r in rows]+sentences)})
  clustered=pd.read_parquet(src/'clustered.parquet');assert np.array_equal(df['id'].to_numpy(),clustered['id'].to_numpy());np.save(d/'labels.npy',clustered['cluster_id'].to_numpy())
  print(topic,len(rows),'exported with frozen legacy_openai and legacy_mpnet')
if __name__=='__main__':main()
