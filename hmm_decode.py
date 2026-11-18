"""Viterbi route decoding and classifier-only comparison on student-test trips."""
import csv,math,os,pathlib,collections,joblib,numpy as np
from train_models import ROOT,FEATURES
from pathsnap_person1 import process_trip
OUT=ROOT/'features'
CAND=pathlib.Path(os.getenv('PATHSNAP_STUDENT_CANDIDATES',str(OUT/'student_test_candidates_aligned.csv')))
MODEL=pathlib.Path(os.getenv('PATHSNAP_MODEL_PATH',str(ROOT/'models'/'extra_trees_candidate_model.joblib')))
DATA=pathlib.Path(os.getenv('PATHSNAP_DATA_ROOT',str(ROOT/'data')))/'student_test'
GAP_BREAK_S=60.0

def continuity(path,arcs,obs_by_idx):
    idxs=sorted(path)
    pairs=[(path[i],path[i+1]) for i in idxs if i+1 in path and i+1 in obs_by_idx and obs_by_idx[i+1]['dt']<=GAP_BREAK_S]
    if not pairs:return float('nan')
    return sum(a==b or arcs[a][1]==arcs[b][0] for a,b in pairs)/len(pairs)

def decode(obs,arcs):
    # Break at missing observations and long GPS gaps; never force transitions over outages.
    segments=[];current=[];last_idx=None
    for item in obs:
        if not item['states']:
            if current: segments.append(current);current=[]
            last_idx=item['idx'];continue
        if current and (item['idx']!=last_idx+1 or item['dt']>GAP_BREAK_S):segments.append(current);current=[]
        current.append(item);last_idx=item['idx']
    if current:segments.append(current)
    decoded={}
    for seq in segments:
        scores=[];backs=[]
        for t,item in enumerate(seq):
            states=item['states'];em=np.log(np.maximum(item['probs'],1e-12))
            if t==0:scores.append(em);backs.append(None);continue
            prev=seq[t-1]['states'];prevscore=scores[-1];new=np.empty(len(states));back=np.empty(len(states),dtype=int)
            for j,b in enumerate(states):
                best=-float('inf');best_i=0
                for i,a in enumerate(prev):
                    if a==b:trans=0.0
                    elif arcs[a][1]==arcs[b][0]:trans=-1.0
                    else:trans=-5.0
                    val=prevscore[i]+trans
                    if val>best:best=val;best_i=i
                new[j]=best+em[j];back[j]=best_i
            scores.append(new);backs.append(back)
        pos=int(np.argmax(scores[-1]));path=[None]*len(seq)
        for t in range(len(seq)-1,-1,-1):
            path[t]=seq[t]['states'][pos]
            if t:pos=int(backs[t][pos])
        for item,aid in zip(seq,path):decoded[item['idx']]=aid
    return decoded

def main():
    bundle=joblib.load(MODEL);model=bundle['model'];feat=bundle['features'];
    grouped=collections.defaultdict(list)
    with CAND.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            vals=[float(r[k] or 0) for k in feat]; grouped[r['trip_id']].append((int(r['obs_idx']),int(r['candidate_arc_id']),vals,int(r['is_route_label']),int(r['label_known']),float(r['dt_s'] or 0)))
    out=OUT/'student_test_hmm_predictions_aligned.csv';summary=[];all_rows=[]
    for tid,rows in grouped.items():
        d=DATA/tid; arcs=[[int(v) for v in l.split()] for l in (d/f'{tid}.arcs').read_text().splitlines() if l.strip()]
        X=np.asarray([r[2] for r in rows],dtype=np.float32);prob=model.predict_proba(X)[:,1]
        byobs=collections.defaultdict(list)
        for r,p in zip(rows,prob):byobs[r[0]].append((r[1],p,r[3],r[4],r[5]))
        obs=[];classifier={};truth={};known={}
        for idx,cands in sorted(byobs.items()):
            states=[x[0] for x in cands];ps=np.asarray([x[1] for x in cands]);obs.append({'idx':idx,'states':states,'probs':ps,'dt':cands[0][4]})
            classifier[idx]=states[int(np.argmax(ps))]
            pos=[x[0] for x in cands if x[2]==1]
            if pos:truth[idx]=pos[0];known[idx]=True
        decoded=decode(obs,arcs)
        _, quality=process_trip(d,'student_test')
        route_labeled=quality['gps_observations']-quality['unlabeled_observations']
        base_hits=sum(classifier[i]==aid for i,aid in truth.items())
        hmm_hits=sum(decoded.get(i)==aid for i,aid in truth.items())
        candidate_misses=max(0,route_labeled-len(truth))
        baseacc=base_hits/max(1,route_labeled);hmmacc=hmm_hits/max(1,route_labeled)
        obs_by_idx={x['idx']:x for x in obs}
        sr={'trip_id':tid,'route_labeled_observations':route_labeled,'candidate_misses':candidate_misses,'classifier_top1_accuracy_all_labels':baseacc,'hmm_top1_accuracy_all_labels':hmmacc,'classifier_continuity':continuity(classifier,arcs,obs_by_idx),'hmm_continuity':continuity(decoded,arcs,obs_by_idx)};summary.append(sr);print(sr,flush=True)
        for i in sorted(decoded):all_rows.append({'trip_id':tid,'obs_idx':i,'classifier_arc_id':classifier[i],'hmm_arc_id':decoded[i],'reference_arc_id':truth.get(i,''),'label_known':int(i in truth)})
    with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    with (OUT/'student_test_hmm_summary_aligned.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    macro={k:float(np.nanmean([r[k] for r in summary])) for k in ['classifier_top1_accuracy_all_labels','hmm_top1_accuracy_all_labels','classifier_continuity','hmm_continuity']}
    total_labels=sum(r['route_labeled_observations'] for r in summary)
    weighted={k:sum(r['route_labeled_observations']*r[k] for r in summary)/max(1,total_labels) for k in ['classifier_top1_accuracy_all_labels','hmm_top1_accuracy_all_labels']}
    aggregate={
        'classifier_weighted_top1_all_route_labeled':weighted['classifier_top1_accuracy_all_labels'],
        'hmm_weighted_top1_all_route_labeled':weighted['hmm_top1_accuracy_all_labels'],
        'classifier_macro_top1_by_trip_all_route_labeled':macro['classifier_top1_accuracy_all_labels'],
        'hmm_macro_top1_by_trip_all_route_labeled':macro['hmm_top1_accuracy_all_labels'],
        'classifier_mean_continuity':macro['classifier_continuity'],
        'hmm_mean_continuity':macro['hmm_continuity'],
    }
    with (OUT/'student_test_hmm_aggregate.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric','value','definition'])
        for key,value in aggregate.items():
            definition='All heuristic route labels; candidate misses count incorrect.' if 'top1' in key else 'Unweighted mean of per-trip adjacent-arc continuity.'
            w.writerow([key,value,definition])
    print('macro means:',macro,'weighted route scores:',weighted)
if __name__=='__main__':main()
