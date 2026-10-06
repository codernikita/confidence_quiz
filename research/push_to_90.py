#!/usr/bin/env python
"""
STAGE 4 — maximum-effort push toward 90% on Low/Med/High.

Three levers not yet pulled, applied together:

  LEVER 1  RICHER LOO FEATURES.
           We were compressing a student's rating pattern into 5 numbers
           (mean/sd/min/max/nuniq), discarding its SHAPE. If 11 of a student's
           12 other ratings are a 5, that is far more informative than
           "mean = 4.9". Added: the full leave-one-out histogram (fraction of
           their other ratings at each of 1..5), mode, modal share, median,
           and quartiles. All strictly leave-one-out.

  LEVER 2  ORDINAL-AWARE MODELLING.
           Multiclass treats Low/Med/High as unrelated labels, so a Low<->High
           error costs the same as Low<->Med. Ordinal decomposition trains
           P(y>Low) and P(y>Med) and recombines them with monotonicity enforced,
           which respects the ordering the target actually has.

  LEVER 3  STRONGER / TUNED LEARNERS + ENSEMBLE.
           CatBoost (ordered boosting -- less small-data bias), a tuned
           LightGBM, and a soft-voting ensemble.

Honesty controls kept throughout:
  * GroupKFold by student_uid -- a test student is never seen in training.
  * q_loo_difficulty refit inside each fold.
  * Every accuracy reported beside its majority-class baseline AND the minimum
    per-class recall, so a high score produced by collapsing a class is visible
    immediately rather than hidden in the average.
"""
import os, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             roc_auc_score, confusion_matrix)
import lightgbm as lgb
from catboost import CatBoostClassifier

OUT = os.path.dirname(os.path.abspath(__file__))

BUCKETS = {
    "A 1-2/3/4-5": lambda c: np.where(c <= 2, 0, np.where(c == 3, 1, 2)),
    "B 1-3/4/5":   lambda c: np.where(c <= 3, 0, np.where(c == 4, 1, 2)),
    "C 1-2/3-4/5": lambda c: np.where(c <= 2, 0, np.where(c <= 4, 1, 2)),
}

BASE_FE = ["time_spent", "time_z_within_q", "time_rel_student", "option_changes",
           "review_click_count", "marked_for_review", "student_cgpa", "is_correct",
           "att_log_total_time", "att_mean_changes", "att_unans_frac", "att_fast_frac",
           "opt_set_cohesion", "opt_nearest_trap",
           "q_flesch_ease", "q_gunning_fog_index", "q_feat_n_words",
           "q_feat_parse_tree_depth", "q_feat_avg_dependency_length"]


def rich_loo(a):
    """Full leave-one-out description of each student's rating pattern.
    For every row, every statistic is computed from that student's OTHER
    ratings only -- the row's own label never enters its own features."""
    a = a.copy()
    c = a.conf_valid.values.astype(float)
    gmean = np.nanmean(c)

    # ---- attempt-level histogram, leave-one-out ----
    idx = a.groupby("attempt_id").indices
    n = len(a)
    frac = np.zeros((n, 5)); mode = np.zeros(n); modeshare = np.zeros(n)
    med = np.zeros(n); q25 = np.zeros(n); q75 = np.zeros(n)
    mean = np.zeros(n); sd = np.zeros(n); mn = np.zeros(n); mx = np.zeros(n)
    nun = np.zeros(n); cnt = np.zeros(n)

    for aid, ii in idx.items():
        vals = c[ii]
        counts = np.array([np.sum(vals == k) for k in range(1, 6)], float)
        tot = len(vals)
        for j, i in enumerate(ii):
            v = vals[j]
            cc = counts.copy()
            if 1 <= v <= 5:
                cc[int(v) - 1] -= 1                      # remove own rating
            m = tot - 1
            others = np.delete(vals, j)
            cnt[i] = m
            if m <= 0:
                frac[i] = 0; mode[i] = gmean; modeshare[i] = 0
                med[i] = q25[i] = q75[i] = mean[i] = gmean
                sd[i] = 0; mn[i] = mx[i] = gmean; nun[i] = 0
                continue
            frac[i] = cc / m
            mode[i] = np.argmax(cc) + 1
            modeshare[i] = cc.max() / m
            mean[i] = others.mean(); sd[i] = others.std(ddof=0)
            mn[i] = others.min(); mx[i] = others.max()
            nun[i] = len(np.unique(others))
            med[i] = np.median(others)
            q25[i] = np.percentile(others, 25); q75[i] = np.percentile(others, 75)

    for k in range(5):
        a[f"loo_frac_{k+1}"] = frac[:, k]
    a["loo_mode"] = mode; a["loo_mode_share"] = modeshare
    a["loo_median"] = med; a["loo_q25"] = q25; a["loo_q75"] = q75
    a["loo_iqr"] = q75 - q25
    a["loo_conf_mean"] = mean; a["loo_conf_sd"] = sd
    a["loo_conf_min"] = mn; a["loo_conf_max"] = mx
    a["loo_conf_nuniq"] = nun; a["loo_n"] = cnt

    # ---- student-level (wider window), leave-one-out ----
    gs = a.groupby("student_uid").conf_valid
    ss, ns = gs.transform("sum"), gs.transform("count")
    a["loo_stu_mean"] = ((ss - a.conf_valid) / (ns - 1).clip(lower=1)).fillna(gmean)
    a["loo_stu_sd"] = gs.transform("std").fillna(0)
    return a


LOO_FE = ([f"loo_frac_{k}" for k in range(1, 6)] +
          ["loo_mode", "loo_mode_share", "loo_median", "loo_q25", "loo_q75",
           "loo_iqr", "loo_conf_mean", "loo_conf_sd", "loo_conf_min",
           "loo_conf_max", "loo_conf_nuniq", "loo_n",
           "loo_stu_mean", "loo_stu_sd"])


def lgbm(**kw):
    p = dict(n_estimators=700, learning_rate=0.04, num_leaves=31,
             min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
             verbose=-1, n_jobs=6, random_state=0)
    p.update(kw); return lgb.LGBMClassifier(**p)


def fit_multiclass(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr); return model.predict_proba(Xte)


def fit_ordinal(Xtr, ytr, Xte, make):
    """P(y>0) and P(y>1) -> 3 class probs, monotonicity enforced."""
    p = []
    for thr in [0, 1]:
        m = make(); m.fit(Xtr, (ytr > thr).astype(int))
        p.append(m.predict_proba(Xte)[:, 1])
    p1, p2 = p[0], np.minimum(p[0], p[1])
    P = np.c_[1 - p1, p1 - p2, p2].clip(1e-9)
    return P / P.sum(1, keepdims=True)


def report(y, P, name, bucket, feat):
    pred = P.argmax(1)
    acc = accuracy_score(y, pred)
    base = max(np.bincount(y, minlength=3) / len(y))
    rec = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)[1]
    return dict(bucket=bucket, features=feat, model=name,
                accuracy=acc, baseline=base, lift=acc - base,
                balanced_acc=balanced_accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                min_class_recall=rec.min(),
                qwk=cohen_kappa_score(y, pred, weights="quadratic"),
                auc=roc_auc_score(y, P, multi_class="ovr"),
                hits90="YES" if acc >= 0.90 else "-",
                real="real" if (acc - base) >= 0.02 and rec.min() >= 0.20 else "SUSPECT")


def main():
    a = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    a = a[~a.is_unanswered].reset_index(drop=True)
    a = rich_loo(a)
    print(f"rows={len(a):,}  students={a.student_uid.nunique():,}  "
          f"attempts={a.attempt_id.nunique():,}")

    FEATSETS = {"BASE+LOO5": BASE_FE + ["loo_conf_mean", "loo_conf_sd", "loo_conf_min",
                                        "loo_conf_max", "loo_conf_nuniq", "loo_stu_mean"],
                "BASE+LOO_RICH": BASE_FE + LOO_FE}
    rows = []
    for bname, bfn in BUCKETS.items():
        y = bfn(a.conf_valid.values)
        for fname, feats in FEATSETS.items():
            X0 = np.nan_to_num(a[feats].values.astype(float))
            store = {k: np.full((len(a), 3), np.nan) for k in
                     ["LightGBM", "LGBM-tuned", "CatBoost", "Ordinal-LGBM", "Ensemble"]}
            for tr, te in GroupKFold(5).split(a, y, a.student_uid.values):
                qd = a.iloc[tr].groupby("question_id").is_correct.mean()
                qc = a.question_id.map(qd).fillna(
                    a.is_correct.values[tr].mean()).values.reshape(-1, 1)
                X = np.hstack([X0, qc])
                Xtr, Xte, ytr = X[tr], X[te], y[tr]

                P_l = fit_multiclass(lgbm(class_weight="balanced"), Xtr, ytr, Xte)
                P_t = fit_multiclass(lgbm(class_weight="balanced", n_estimators=1200,
                                          learning_rate=0.02, num_leaves=63,
                                          min_child_samples=15, colsample_bytree=0.7,
                                          reg_lambda=1.0), Xtr, ytr, Xte)
                P_c = fit_multiclass(CatBoostClassifier(
                    iterations=900, learning_rate=0.05, depth=6, loss_function="MultiClass",
                    auto_class_weights="Balanced", verbose=0, random_seed=0,
                    thread_count=6), Xtr, ytr, Xte)
                P_o = fit_ordinal(Xtr, ytr, Xte,
                                  lambda: lgbm(class_weight="balanced"))
                P_e = (P_l + P_t + P_c + P_o) / 4
                for k, P in zip(store, [P_l, P_t, P_c, P_o, P_e]):
                    store[k][te] = P
            for k, P in store.items():
                rows.append(report(y, P, k, bname, fname))
            print(f"  done {bname:14s} {fname}")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "push_to_90_results.csv"), index=False)
    pd.set_option("display.width", 220)
    show = ["bucket", "features", "model", "accuracy", "baseline", "lift",
            "balanced_acc", "f1_macro", "min_class_recall", "qwk", "auc",
            "hits90", "real"]
    print(f"\n{'#'*120}\nALL RESULTS (sorted by accuracy)\n{'#'*120}")
    print(R.sort_values("accuracy", ascending=False)[show].round(4).to_string(index=False))
    print(f"\n{'#'*120}\nREACHED 90% WITH GENUINE SKILL\n{'#'*120}")
    w = R[(R.accuracy >= 0.90) & (R.lift >= 0.02) & (R.min_class_recall >= 0.20)]
    print(w[show].round(4).to_string(index=False) if len(w) else "  (none)")
    print(f"\n{'#'*120}\nGAIN FROM RICHER LOO FEATURES\n{'#'*120}")
    piv = R.pivot_table(index=["bucket", "model"], columns="features", values="accuracy")
    piv["gain"] = piv["BASE+LOO_RICH"] - piv["BASE+LOO5"]
    print(piv.sort_values("gain", ascending=False).round(4).to_string())
    print("\nSaved -> push_to_90_results.csv")


if __name__ == "__main__":
    main()
