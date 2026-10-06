"""Does an EASY/MEDIUM/HARD calibration triple beat a random triple?
Tested per-quiz, 12 shuffles, scored on non-calibration rows only."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
from push_to_90 import BASE_FE
OUT=os.path.dirname(os.path.abspath(__file__))
DEAD=["q_feat_parse_tree_depth","att_fast_frac"]
NUM=[f for f in BASE_FE if f not in DEAD]
NSH=12

def calib_feats(sub,mask,gm):
    out=np.full((len(sub),9),np.nan); c=sub.conf_valid.values.astype(float)
    for aid,ii in sub.groupby("attempt_id").indices.items():
        cal=ii[mask[ii]]
        if len(cal)==0: continue
        v=c[cal]; out[ii,:5]=[np.mean(v==k) for k in range(1,6)]
        out[ii,5:]=[v.mean(),v.std(ddof=0),v.min(),v.max()]
    out[:,:5]=np.nan_to_num(out[:,:5])
    for j in range(5,9): out[:,j]=np.where(np.isnan(out[:,j]),gm,out[:,j])
    return out

def score(sub,mask,y,Xn,gm):
    CF=calib_feats(sub,mask,gm); P=np.full((len(sub),3),np.nan)
    for tr,te in GroupKFold(5).split(sub,y,sub.student_uid.values):
        qd=sub.iloc[tr].groupby("question_id").is_correct.mean()
        qc=sub.question_id.map(qd).fillna(sub.is_correct.values[tr].mean()).values.reshape(-1,1)
        X=np.hstack([Xn,CF,qc]); tr2,te2=tr[~mask[tr]],te[~mask[te]]
        if len(tr2)<60 or len(te2)<10: continue
        m=lgb.LGBMClassifier(n_estimators=250,learning_rate=.06,num_leaves=15,
            min_child_samples=15,subsample=.8,colsample_bytree=.8,
            class_weight="balanced",verbose=-1,n_jobs=6,random_state=0)
        m.fit(X[tr2],y[tr2]); P[te2]=m.predict_proba(X[te2])
    ok=np.isfinite(P).all(1)
    if ok.sum()<50: return None
    pr,yy=P[ok].argmax(1),y[ok]
    return accuracy_score(yy,pr),balanced_accuracy_score(yy,pr),roc_auc_score(yy,P[ok],multi_class="ovr")

a=pd.read_pickle(os.path.join(OUT,"cleaned.pkl")); a=a[~a.is_unanswered]
QZ=[("DMS_QUIZ_second_APPENDED","DMS-2"),("Quiz_Attempts_ELC_1_APPENDED","ELC-1"),
    ("DBMS_140726_Quiz_Attempts_Report","DBMS")]
print(f"{'quiz':7s} {'strategy':22s} {'acc':>7s} {'sd':>7s} {'bal':>7s} {'auc':>7s}")
print("-"*62)
for src,lab in QZ:
    sub=a[a.__src==src].reset_index(drop=True)
    if len(sub)<300: continue
    c=sub.conf_valid.values; y=np.where(c<=2,0,np.where(c<=4,1,2))
    gm=np.nanmean(c); Xn=np.nan_to_num(sub[NUM].values.astype(float))
    qdiff=sub.groupby("question_id").is_correct.mean().sort_values()
    nq=len(qdiff)
    hard=list(qdiff.index[:max(1,nq//3)])          # lowest accuracy = hardest
    med =list(qdiff.index[nq//3:2*nq//3])
    easy=list(qdiff.index[2*nq//3:])
    strategies={}
    # EASY/MED/HARD: one from each tercile
    res={}
    for name in ["random-3","easy/med/hard","3 easiest","3 hardest","3 medium"]:
        accs=[];bals=[];aucs=[]
        for s in range(NSH):
            rng=np.random.RandomState(s)
            mask=np.zeros(len(sub),bool)
            for aid,ii in sub.groupby("attempt_id").indices.items():
                qs=sub.question_id.values[ii]
                if name=="random-3":
                    nt=max(0,min(3,len(ii)-2))
                    pick=rng.choice(len(ii),nt,replace=False) if nt else np.array([],dtype=int)
                else:
                    pool={"easy/med/hard":[easy,med,hard],"3 easiest":[easy,easy,easy],
                          "3 hardest":[hard,hard,hard],"3 medium":[med,med,med]}[name]
                    want=[]
                    for grp in pool:
                        cand=[j for j,q in enumerate(qs) if q in grp and j not in want]
                        if cand: want.append(rng.choice(cand))
                    pick=np.array(want,dtype=int)
                    if len(pick)>len(ii)-2: pick=pick[:max(0,len(ii)-2)]
                if len(pick): mask[ii[pick]]=True
            r=score(sub,mask,y,Xn,gm)
            if r: accs.append(r[0]);bals.append(r[1]);aucs.append(r[2])
        if accs:
            print(f"{lab:7s} {name:22s} {np.mean(accs):7.4f} {np.std(accs):7.4f} "
                  f"{np.mean(bals):7.4f} {np.mean(aucs):7.4f}",flush=True)
    print()
