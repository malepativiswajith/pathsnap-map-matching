"""Fit provisional LR on train trips and evaluate once on student_test.
Student-test route labels are used only offline for evaluation, never as model features.
"""
import csv,pathlib,random,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,log_loss
from train_models import ROOT,FEATURES,SEED

OUT=ROOT/'features';TRAIN=OUT/'train_candidates.csv';TEST=OUT/'student_test_candidates.csv'
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
 top_hits=[];known_without_candidate=0
 for rec in byobs.values():
  idx=rec['indices'];true=rec['true_arc']
  if true is None:continue
  if not any(at[i]==true for i in idx):
   known_without_candidate+=1;top_hits.append(0);continue
  chosen=max(idx,key=lambda i:prob[i]);top_hits.append(int(at[chosen]==true))
 cov_obs=sum(r['true_arc'] is not None for r in byobs.values()); nobs=len(byobs)
 result={'model':'Logistic Regression C=1','train_trips':len(set(g)),'student_test_trips':len(set(gt)),'train_rows_after_negative_sampling':len(keep),'student_candidate_rows':len(yt),'student_observations_with_candidates':nobs,'route_label_coverage':cov_obs,'candidate_misses':known_without_candidate,'candidate_miss_rate':known_without_candidate/max(1,cov_obs),'top1_accuracy_all_route_labeled_obs':float(np.mean(top_hits)) if top_hits else 0.0,'candidate_accuracy':accuracy_score(yt,hard),'candidate_macro_f1':f1_score(yt,hard,average='macro',zero_division=0),'candidate_log_loss':log_loss(yt,prob,labels=[0,1]),'train_seconds':fit_s,'inference_ms_per_observation':1000*infer_s/max(1,nobs)}
 OUT.mkdir(exist_ok=True)
 with (OUT/'student_test_metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(result));w.writeheader();w.writerow(result)
 print(result)
if __name__=='__main__':main()
