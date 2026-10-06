#!/usr/bin/env python
"""
CALIBRATION SWEEP — single-quiz, randomised assignment, k = 1..n-3
==================================================================

Fixes two confounds in the earlier pooled test:

  CONFOUND 1  Pooling all 10 quizzes mixes subjects with different difficulty
              and different numbers of questions, so "k=5" meant different
              fractions of the quiz in different workbooks.
              FIX: run within ONE quiz at a time.

  CONFOUND 2  "first-k" is confounded with question identity, because question
              order is fixed per workbook (only 1-4 distinct orders exist). The
              calibration block was therefore always the SAME questions, and any
              measured effect could be a property of those particular items.
              FIX: assign calibration questions at random, independently per
              attempt, and repeat over N_SHUFFLE seeds. Reported numbers are the
              mean +/- sd across shuffles, so no single question can drive them.

Scoring is on the NON-calibration rows only. Model is LightGBM: these per-quiz
subsets are 700-3,000 rows, far too small to trust a neural net, and boosting is
the stable choice at this scale.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score)
import lightgbm as lgb
from push_to_90 import BASE_FE

OUT = os.path.dirname(os.path.abspath(__file__))
DEAD = ["q_feat_parse_tree_depth", "att_fast_frac"]
NUM_FE = [f for f in BASE_FE if f not in DEAD]
N_SHUFFLE = 8

QUIZZES = [
    ("DBMS_140726_Quiz_Attempts_Report", "DBMS (15 q/attempt, 49 attempts)"),
    ("DMS_QUIZ_second_APPENDED",         "DMS-2 (13 q/attempt, 146 attempts)"),
    ("Quiz_Attempts_ELC_1_APPENDED",     "ELC-1 (11 q/attempt, 276 attempts)"),
]


def calib_features(sub, calib_mask, gmean):
    """Confidence summary built ONLY from that attempt's calibration rows."""
    n = len(sub)
    out = np.full((n, 9), np.nan)
    conf = sub.conf_valid.values.astype(float)
    for aid, ii in sub.groupby("attempt_id").indices.items():
        cal = ii[calib_mask[ii]]
        if len(cal) == 0:
            continue
        v = conf[cal]
        out[ii, :5] = [np.mean(v == k) for k in range(1, 6)]
        out[ii, 5:] = [v.mean(), v.std(ddof=0), v.min(), v.max()]
    out[:, :5] = np.nan_to_num(out[:, :5])
    for j in range(5, 9):
        out[:, j] = np.where(np.isnan(out[:, j]), gmean, out[:, j])
    return out


def run_quiz(sub, label):
    sub = sub.reset_index(drop=True)
    conf = sub.conf_valid.values
    y = np.where(conf <= 2, 0, np.where(conf <= 4, 1, 2))
    gmean = np.nanmean(conf)
    Xnum = np.nan_to_num(sub[NUM_FE].values.astype(float))
    sizes = sub.groupby("attempt_id").size()
    nq = int(sizes.median())
    print(f"\n{'='*100}\n{label}   rows={len(sub):,}  attempts={sub.attempt_id.nunique()}  "
          f"students={sub.student_uid.nunique()}  median q/attempt={nq}")
    print(f"class balance Low/Med/High = "
          f"{np.round(np.bincount(y, minlength=3)/len(y), 3)}")
    print(f"{'='*100}")

    rows = []
    for k in range(0, max(1, nq - 2)):
        accs, bals, f1s, qwks, aucs, bases, ns = [], [], [], [], [], [], []
        for seed in range(1 if k == 0 else N_SHUFFLE):
            rng = np.random.RandomState(1000 * k + seed)
            mask = np.zeros(len(sub), bool)
            if k > 0:
                for aid, ii in sub.groupby("attempt_id").indices.items():
                    take = min(k, max(0, len(ii) - 2))       # always leave >=2 to score
                    if take > 0:
                        mask[rng.choice(ii, take, replace=False)] = True
            CF = calib_features(sub, mask, gmean)
            P = np.full((len(sub), 3), np.nan)
            for tr, te in GroupKFold(5).split(sub, y, sub.student_uid.values):
                qd = sub.iloc[tr].groupby("question_id").is_correct.mean()
                qc = sub.question_id.map(qd).fillna(
                    sub.is_correct.values[tr].mean()).values.reshape(-1, 1)
                X = np.hstack([Xnum, CF, qc])
                tr2, te2 = tr[~mask[tr]], te[~mask[te]]
                if len(tr2) < 60 or len(te2) < 10:
                    continue
                m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.06,
                                       num_leaves=15, min_child_samples=15,
                                       subsample=0.8, colsample_bytree=0.8,
                                       class_weight="balanced", verbose=-1,
                                       n_jobs=6, random_state=0)
                m.fit(X[tr2], y[tr2]); P[te2] = m.predict_proba(X[te2])
            ok = np.isfinite(P).all(1)
            if ok.sum() < 50:
                continue
            pr, yy = P[ok].argmax(1), y[ok]
            accs.append(accuracy_score(yy, pr))
            bals.append(balanced_accuracy_score(yy, pr))
            f1s.append(f1_score(yy, pr, average="macro", zero_division=0))
            qwks.append(cohen_kappa_score(yy, pr, weights="quadratic"))
            try:
                aucs.append(roc_auc_score(yy, P[ok], multi_class="ovr"))
            except Exception:
                aucs.append(np.nan)
            bases.append(max(np.bincount(yy, minlength=3) / len(yy)))
            ns.append(int(ok.sum()))
        if not accs:
            continue
        rows.append(dict(quiz=label, k=k, n_scored=int(np.mean(ns)),
                         acc=np.mean(accs), acc_sd=np.std(accs),
                         baseline=np.mean(bases), lift=np.mean(accs) - np.mean(bases),
                         bal_acc=np.mean(bals), f1=np.mean(f1s),
                         qwk=np.mean(qwks), auc=np.nanmean(aucs)))
        r = rows[-1]
        print(f"  k={k:2d}  acc={r['acc']:.4f} +/-{r['acc_sd']:.4f}  "
              f"base={r['baseline']:.4f}  lift={r['lift']:+.4f}  "
              f"bal={r['bal_acc']:.4f}  auc={r['auc']:.4f}  n={r['n_scored']:,}",
              flush=True)
    return rows


def main():
    a = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    a = a[~a.is_unanswered]
    print(f"N_SHUFFLE = {N_SHUFFLE} random calibration assignments per k")
    print("Reported values are means across shuffles, so no single question "
          "can drive the result.")
    allrows = []
    for src, label in QUIZZES:
        sub = a[a.__src == src]
        if len(sub) < 300:
            print(f"\n[skip] {label}: only {len(sub)} rows")
            continue
        allrows += run_quiz(sub, label)

    R = pd.DataFrame(allrows)
    R.to_csv(os.path.join(OUT, "calibration_sweep.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{'#'*100}\nMARGINAL VALUE OF EACH EXTRA CALIBRATION QUESTION\n{'#'*100}")
    for q in R.quiz.unique():
        d = R[R.quiz == q].sort_values("k").reset_index(drop=True)
        d["gain_vs_prev"] = d.acc.diff()
        d["gain_vs_k0"] = d.acc - d.acc.iloc[0]
        print(f"\n{q}")
        print(d[["k", "n_scored", "acc", "acc_sd", "baseline", "lift",
                 "bal_acc", "auc", "gain_vs_prev", "gain_vs_k0"]].round(4).to_string(index=False))
    print("\nSaved -> calibration_sweep.csv")


if __name__ == "__main__":
    main()
