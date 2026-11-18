"""Fit provisional LR on train trips and evaluate once on student_test.
Student-test route labels are used only offline for evaluation, never as model features.
"""
import csv,os,pathlib,random,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,log_loss
from train_models import ROOT,FEATURES,SEED
from pathsnap_person1 import process_trip

OUT=ROOT/'features'
DATA_ROOT=pathlib.Path(os.getenv('PATHSNAP_DATA_ROOT',str(ROOT/'data')))
TRAIN=pathlib.Path(os.getenv('PATHSNAP_TRAIN_CANDIDATES',str(OUT/'train_candidates_aligned.csv')))
TEST=pathlib.Path(os.getenv('PATHSNAP_STUDENT_CANDIDATES',str(OUT/'student_test_candidates_aligned.csv')))
def load(path):
 X=[];y=[];groups=[];obs=[];arcs=[]
 with path.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['label_known']!='1':continue
   X.append([float(r[k] or 0) for k in FEATURES]);y.append(int(r['is_route_label']));groups.append(r['trip_id']);obs.append((r['trip_id'],int(r['obs_idx'])));arcs.append(int(r['candidate_arc_id']))
 return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int8),np.asarray(groups),np.asarray(obs,dtype=object),np.asarray(arcs)
def sample(X,y,obs):
 rng=random.Random(SEED); bins={}
 for i,k in enumerate(obs):bins.setdefault(tuple(k),[]).append(i)
 keep=[]
 for ids in bins.values():
  p=[i for i in ids if y[i]==1];n=[i for i in ids if y[i]==0]
  keep.extend(p);keep.extend(rng.sample(n,min(4,len(n))))
 return np.asarray(keep,dtype=int)
def main():
 if not TRAIN.exists():raise SystemExit(f'Missing training candidates: {TRAIN}')
 if not TEST.exists():raise SystemExit(f'Missing student-test candidates: {TEST}')
 X,y,g,o,a=load(TRAIN); keep=sample(X,y,o)
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=300,solver='lbfgs'))
 start=time.perf_counter();model.fit(X[keep],y[keep]);fit_s=time.perf_counter()-start
 Xt,yt,gt,ot,at=load(TEST)
 start=time.perf_counter();prob=model.predict_proba(Xt)[:,1];infer_s=time.perf_counter()-start
 hard=prob>=.5
 byobs={}
 for i,key in enumerate(ot):
  key=tuple(key);rec=byobs.setdefault(key,{'indices':[],'true_arc':None})
  rec['indices'].append(i)
  if yt[i]==1:rec['true_arc']=at[i]
 top_hits=[]
 for rec in byobs.values():
  idx=rec['indices'];true=rec['true_arc']
  if true is None:
   top_hits.append(0);continue
  chosen=max(idx,key=lambda i:prob[i]);top_hits.append(int(at[chosen]==true))
 route_labeled=0;total_gps=0
 for folder in sorted(p for p in (DATA_ROOT/'student_test').iterdir() if p.is_dir()):
  _,summary=process_trip(folder,'student_test')
  route_labeled+=summary['gps_observations']-summary['unlabeled_observations']
  total_gps+=summary['gps_observations']
 positive_present=sum(r['true_arc'] is not None for r in byobs.values())
 misses=max(0,route_labeled-positive_present)
 result={'model':'Logistic Regression C=1','train_trips':len(set(g)),'student_test_trips':len(set(gt)),'train_rows_after_negative_sampling':len(keep),'student_candidate_rows':len(yt),'student_observations_with_candidates':len(byobs),'route_labeled_observations':route_labeled,'route_label_coverage_rate':route_labeled/max(1,total_gps),'candidate_misses':misses,'candidate_miss_rate':misses/max(1,route_labeled),'top1_accuracy_all_route_labeled_obs':sum(top_hits)/max(1,route_labeled),'candidate_accuracy':accuracy_score(yt,hard),'candidate_macro_f1':f1_score(yt,hard,average='macro',zero_division=0),'candidate_log_loss':log_loss(yt,prob,labels=[0,1]),'train_seconds':fit_s,'inference_ms_per_observation':1000*infer_s/max(1,len(byobs))}
 OUT.mkdir(exist_ok=True)
 with (OUT/'student_test_metrics_aligned.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(result));w.writeheader();w.writerow(result)
 print(result)
if __name__=='__main__':main()
