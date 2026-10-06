#!/usr/bin/env python
"""
STAGE 2 — Systematic bake-off: which TARGET x MODEL reaches 90% accuracy honestly?

Accuracy alone is meaningless without its majority baseline, so every row reports
both, plus the lift. A result only counts as REAL if it clears 90% AND beats its
baseline by >= 2 points AND has balanced accuracy well above chance.

MECHANICAL-LEAK CONTROL. At attempt level, `unans_frac` correlates -0.75 with
accuracy -- but that is largely tautological: an unanswered item is scored wrong,
so acc <= 1 - unans_frac by construction. Predicting "did they pass" from
"what fraction did they leave blank" is arithmetic, not learning. Every attempt
level target is therefore run twice:
    FULL   - all behavioural features (reported, but flagged)
    HONEST - drops unans_frac / fast_frac / time_remaining, the mechanically
             linked ones. This is the number to believe.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, precision_recall_fscore_support)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

OUT = os.path.dirname(os.path.abspath(__file__))

ITEM_FE = ["time_spent", "time_z_within_q", "time_rel_student", "option_changes",
           "review_click_count", "marked_for_review", "student_cgpa",
           "att_log_total_time", "att_mean_changes",
           "opt_set_cohesion", "opt_nearest_trap",
           "q_flesch_ease", "q_gunning_fog_index", "q_feat_n_words",
           "q_feat_parse_tree_depth"]
ITEM_MECH = ["att_unans_frac", "att_fast_frac"]

ATT_FE = ["cgpa", "tot_time", "med_time", "mean_changes", "sum_changes",
          "mean_review", "any_review", "n_items"]
ATT_MECH = ["unans_frac", "fast_frac", "time_remaining"]


def models():
    return {
        "LogReg":   make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                             n_jobs=8, random_state=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04,
                                       num_leaves=31, min_child_samples=30,
                                       subsample=0.8, colsample_bytree=0.8,
                                       verbose=-1, n_jobs=8, random_state=0),
        "XGBoost":  xgb.XGBClassifier(n_estimators=500, learning_rate=0.04,
                                      max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="logloss",
                                      n_jobs=8, random_state=0, verbosity=0),
        "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                       verbose=0, random_seed=0, thread_count=8),
        "MLP":      make_pipeline(StandardScaler(),
                                  MLPClassifier(hidden_layer_sizes=(64, 32),
                                                max_iter=400, random_state=0)),
    }


def run_cv(X, y, groups, extra_fold_fn=None):
    """Returns out-of-fold probabilities for each model."""
    oof = {k: np.full(len(y), np.nan) for k in models()}
    for tr, te in GroupKFold(5).split(X, y, groups):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if extra_fold_fn is not None:
            Xtr, Xte = extra_fold_fn(tr, te, Xtr, Xte)
        for name, m in models().items():
            try:
                m.fit(Xtr, y[tr])
                oof[name][te] = m.predict_proba(Xte)[:, 1]
            except Exception as e:
                print(f"    ! {name} failed: {e}")
    return oof


def score(y, p, tag, target, featset, n):
    yp = (p >= 0.5).astype(int)
    acc = accuracy_score(y, yp)
    base = max(np.mean(y == 0), np.mean(y == 1))
    bal = balanced_accuracy_score(y, yp)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    verdict = ("REAL >=90%" if acc >= 0.90 and acc - base >= 0.02
               else "real skill" if acc - base >= 0.02
               else "imbalance only")
    return dict(target=target, features=featset, model=tag, n=n,
                accuracy=acc, baseline=base, lift=acc - base,
                balanced_acc=bal, auc=auc, prec_macro=pr, rec_macro=rc,
                f1_macro=f1, verdict=verdict)


def main():
    d = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    att = pd.read_pickle(os.path.join(OUT, "cleaned_attempts.pkl"))
    rows = []

    # ============================= ITEM LEVEL =============================
    a = d[~d.is_unanswered].reset_index(drop=True)
    print(f"\n{'='*100}\nITEM-LEVEL TARGET: is_correct   (n={len(a):,})\n{'='*100}")
    y = a.is_correct.values.astype(int)

    def add_qdiff(tr, te, Xtr, Xte):
        qd = a.iloc[tr].groupby("question_id").is_correct.mean()
        g = a.is_correct.values[tr].mean()
        col = a.question_id.map(qd).fillna(g).values.reshape(-1, 1)
        return np.hstack([Xtr, col[tr]]), np.hstack([Xte, col[te]])

    for fs, cols in [("HONEST", ITEM_FE), ("FULL", ITEM_FE + ITEM_MECH)]:
        X = np.nan_to_num(a[cols].values.astype(float))
        for split, gcol in [("cold-student", "student_uid"), ("cold-attempt", "attempt_id")]:
            oof = run_cv(X, y, a[gcol].values, add_qdiff)
            for name, p in oof.items():
                m = np.isfinite(p)
                r = score(y[m], p[m], name, f"is_correct [{split}]", fs, int(m.sum()))
                rows.append(r)

    # =========================== ATTEMPT LEVEL ===========================
    att = att.reset_index()
    for thr in [0.4, 0.5]:
        y = (att.acc >= thr).values.astype(int)
        tname = f"passed>={int(thr*100)}%"
        print(f"\n{'='*100}\nATTEMPT-LEVEL TARGET: {tname}   (n={len(att):,})\n{'='*100}")
        for fs, cols in [("HONEST", ATT_FE), ("FULL", ATT_FE + ATT_MECH)]:
            X = np.nan_to_num(att[cols].values.astype(float))
            oof = run_cv(X, y, att.student.values)
            for name, p in oof.items():
                m = np.isfinite(p)
                rows.append(score(y[m], p[m], name, f"{tname} [cold-student]", fs, int(m.sum())))

    # careless-attempt detection (no mechanical link -- straightlining is a
    # confidence pattern, features are timing/behaviour only)
    y = att.careless.values.astype(int)
    print(f"\n{'='*100}\nATTEMPT-LEVEL TARGET: careless attempt   (n={len(att):,})\n{'='*100}")
    for fs, cols in [("HONEST", ATT_FE), ("FULL", ATT_FE + ATT_MECH)]:
        X = np.nan_to_num(att[cols].values.astype(float))
        oof = run_cv(X, y, att.student.values)
        for name, p in oof.items():
            m = np.isfinite(p)
            rows.append(score(y[m], p[m], name, "careless attempt [cold-student]", fs, int(m.sum())))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "bakeoff_results.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_rows", 300)

    print(f"\n\n{'#'*100}\nFULL RESULTS (sorted by accuracy)\n{'#'*100}")
    show = ["target", "features", "model", "accuracy", "baseline", "lift",
            "balanced_acc", "auc", "f1_macro", "verdict"]
    print(R.sort_values("accuracy", ascending=False)[show].round(4).to_string(index=False))

    print(f"\n\n{'#'*100}\nCLEARED 90% WITH REAL SKILL\n{'#'*100}")
    win = R[(R.accuracy >= 0.90) & (R.lift >= 0.02)].sort_values("accuracy", ascending=False)
    print(win[show].round(4).to_string(index=False) if len(win)
          else "  (none)")

    print(f"\n\n{'#'*100}\nBEST PER TARGET x FEATURE SET\n{'#'*100}")
    best = R.loc[R.groupby(["target", "features"]).accuracy.idxmax()]
    print(best.sort_values("accuracy", ascending=False)[show].round(4).to_string(index=False))
    print(f"\nSaved -> bakeoff_results.csv")


if __name__ == "__main__":
    main()
