"""Does a student's CONFIDENCE STYLE transfer across subjects?
If yes -> a fixed, subject-independent calibration block is viable.
If no  -> calibration items must come from the same subject as the quiz."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
from push_to_90 import BASE_FE
OUT=os.path.dirname(os.path.abspath(__file__))
DEAD=["q_feat_parse_tree_depth","att_fast_frac"]; NUM=[f for f in BASE_FE if f not in DEAD]

a=pd.read_pickle(os.path.join(OUT,"cleaned.pkl")); a=a[~a.is_unanswered]
per=a.groupby("student_uid").__src.nunique()
multi=per[per>=2].index
m=a[a.student_uid.isin(multi)]
print(f"cross-quiz students={len(multi)}  rows={len(m):,}")

# ---- 1. Does mean confidence transfer between subjects? ----
piv=m.groupby(["student_uid","__src"]).conf_valid.agg(["mean","std","count"]).reset_index()
piv=piv[piv["count"]>=4]
pairs=[]
for sid,g in piv.groupby("student_uid"):
    if len(g)<2: continue
    g=g.sort_values("__src")
    pairs.append((g["mean"].iloc[0], g["mean"].iloc[1],
                  g["std"].iloc[0], g["std"].iloc[1]))
P=pd.DataFrame(pairs,columns=["meanA","meanB","sdA","sdB"]).dropna()
print(f"\n=== 1. CROSS-SUBJECT transfer of confidence style (n={len(P)} students) ===")
print(f"  corr(mean conf in subject A, mean conf in subject B): "
      f"pearson={pearsonr(P.meanA,P.meanB)[0]:.4f}  spearman={spearmanr(P.meanA,P.meanB).statistic:.4f}")
print(f"  corr(sd   in A, sd   in B): pearson={pearsonr(P.sdA,P.sdB)[0]:.4f}")

# ---- 2. Benchmark: WITHIN-subject, 3 items vs the rest ----
rng=np.random.RandomState(0); wr=[]
for aid,g in a.groupby("attempt_id"):
    if len(g)<8: continue
    v=g.conf_valid.values; idx=rng.permutation(len(v))
    wr.append((v[idx[:3]].mean(), v[idx[3:]].mean()))
Wd=pd.DataFrame(wr,columns=["cal3","rest"]).dropna()
print(f"\n=== 2. WITHIN-subject benchmark (n={len(Wd)} attempts) ===")
print(f"  corr(mean of 3 random items, mean of remaining items): "
      f"pearson={pearsonr(Wd.cal3,Wd.rest)[0]:.4f}  spearman={spearmanr(Wd.cal3,Wd.rest).statistic:.4f}")

# ---- 3. Actually PREDICT subject-B confidence from subject-A calibration ----
print(f"\n=== 3. Predict subject-B Low/Med/High using subject-A confidence as calibration ===")
first_src=m.groupby("student_uid").__src.first()
rows=[]
for sid,g in m.groupby("student_uid"):
    srcs=sorted(g.__src.unique())
    if len(srcs)<2: continue
    A,B=srcs[0],srcs[1]
    ga,gb=g[g.__src==A],g[g.__src==B]
    if len(ga)<4 or len(gb)<4: continue
    va=ga.conf_valid.values
    stats=dict(calA_mean=va.mean(),calA_sd=va.std(),calA_min=va.min(),calA_max=va.max(),
               **{f"calA_frac_{k}":np.mean(va==k) for k in range(1,6)})
    for _,r in gb.iterrows():
        rows.append({**stats, **{f:r[f] for f in NUM}, "y":r.conf_valid,
                     "student_uid":sid, "is_correct":r.is_correct,
                     "question_id":r.question_id})
D=pd.DataFrame(rows)
if len(D)>200:
    y=np.where(D.y<=2,0,np.where(D.y<=4,1,2))
    CAL=[c for c in D.columns if c.startswith("calA_")]
    for tag,cols in [("behaviour only",NUM),("behaviour + subject-A calibration",NUM+CAL)]:
        P3=np.full((len(D),3),np.nan)
        for tr,te in GroupKFold(5).split(D,y,D.student_uid.values):
            qd=D.iloc[tr].groupby("question_id").is_correct.mean()
            qc=D.question_id.map(qd).fillna(D.is_correct.values[tr].mean()).values.reshape(-1,1)
            X=np.hstack([np.nan_to_num(D[cols].values.astype(float)),qc])
            mm=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=15,
                min_child_samples=15,class_weight="balanced",verbose=-1,n_jobs=6,random_state=0)
            mm.fit(X[tr],y[tr]); P3[te]=mm.predict_proba(X[te])
        ok=np.isfinite(P3).all(1); pr,yy=P3[ok].argmax(1),y[ok]
        base=max(np.bincount(yy,minlength=3)/len(yy))
        print(f"  {tag:36s} acc={accuracy_score(yy,pr):.4f} base={base:.4f} "
              f"lift={accuracy_score(yy,pr)-base:+.4f} bal={balanced_accuracy_score(yy,pr):.4f} "
              f"auc={roc_auc_score(yy,P3[ok],multi_class='ovr'):.4f}  n={int(ok.sum()):,}")
else:
    print(f"  insufficient paired rows ({len(D)})")
