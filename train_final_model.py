"""Fit the selected Logistic Regression on all training trips and save the pipeline."""
import pathlib,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train_models import ROOT,FEATURES,load,downsample,SEED
X,y,g,obs,_=load();Xs,ys,_=downsample(X,y,g,obs,SEED)
model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=300,solver='lbfgs'))
model.fit(Xs,ys);out=ROOT/'models'/'logistic_candidate_model.joblib';out.parent.mkdir(exist_ok=True);joblib.dump({'model':model,'features':FEATURES,'C':1.0},out)
print(f'Saved {out}; trained on {len(set(g))} training trips and {len(ys):,} sampled candidate rows.')
