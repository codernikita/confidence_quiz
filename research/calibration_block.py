#!/usr/bin/env python
"""
CALIBRATION-BLOCK DESIGN TEST
=============================

Proposal under test: the quiz opens with k "calibration" questions where the
student DOES rate their confidence. Those k ratings are then used to build the
leave-one-out style features that predict confidence on the remaining questions,
where no rating is collected.

This differs from the earlier SUBSET-k ablation in a way that matters:
  * SUBSET-k drew k random other ratings, redrawn per row.
  * Here ONE fixed calibration block per attempt serves every later question,
    which is what a real quiz would do -- and is strictly harder.

Evaluated ONLY on the non-calibration questions, because the calibration ones
have a known rating and need no prediction.

Three calibration-selection strategies:
  FIRST-k   the first k questions in the attempt (simplest to build)
  RANDOM-k  k random questions from the attempt
  BEST-k    the k questions ranked globally as the most informative calibrators,
            where "informative" = how strongly confidence on that question
            correlates with the student's mean confidence elsewhere. This is the
            data-driven version of "psychologically chosen" calibration items.

BEST-k ranking is computed on the TRAINING FOLD ONLY, so the choice of
calibration questions never sees test-set data.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             roc_auc_score)
import torch, torch.nn as nn
from push_to_90 import BASE_FE

OUT = os.path.dirname(os.path.abspath(__file__))
CATS = ["q_type", "q_blooms_level", "q_difficulty", "__src"]
DEAD = ["q_feat_parse_tree_depth", "att_fast_frac"]
NUM_FE = [f for f in BASE_FE if f not in DEAD]


class CatEmbNet(nn.Module):
    """Winning Stage-6 architecture: adam, emb=8, (128,64), lr=5e-3, do=0.4."""
    def __init__(self, n_num, levels, emb_dim=8, hidden=(128, 64), dropout=0.4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in levels])
        prev, layers = n_num + emb_dim * len(levels), []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        return self.net(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1))


def fit_predict(Xtr, ctr, ytr, Xte, cte, levels, epochs=60, seed=0):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ct, ce = torch.tensor(ctr, dtype=torch.long), torch.tensor(cte, dtype=torch.long)
    yt = torch.tensor(ytr, dtype=torch.long)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32)
    net = CatEmbNet(Xtr.shape[1], levels)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss(weight=w)
    n = len(ytr)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            if len(b) < 2: continue
            opt.zero_grad(); lf(net(xt[b], ct[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(xe, ce), 1).numpy()


def calib_features(a, calib_mask, gmean):
    """Build confidence features from ONLY the calibration rows of each attempt.
    Rows inside the calibration block get NaN -> they are excluded from scoring."""
    n = len(a)
    out = np.full((n, 9), np.nan)
    conf = a.conf_valid.values.astype(float)
    for aid, ii in a.groupby("attempt_id").indices.items():
        cal = ii[calib_mask[ii]]
        if len(cal) == 0:
            continue
        v = conf[cal]
        counts = np.array([np.sum(v == k) for k in range(1, 6)], float) / len(v)
        stats = np.array([v.mean(), v.std(ddof=0), v.min(), v.max()])
        out[ii, :5] = counts
        out[ii, 5:] = stats
    out[:, :5] = np.nan_to_num(out[:, :5])
    for j in range(5, 9):
        out[:, j] = np.where(np.isnan(out[:, j]), gmean, out[:, j])
    return out


CAL_NAMES = ["cal_frac_1", "cal_frac_2", "cal_frac_3", "cal_frac_4", "cal_frac_5",
             "cal_mean", "cal_sd", "cal_min", "cal_max"]


def pick_calibration(a, k, strategy, train_idx, rng):
    """Return a boolean mask marking calibration rows."""
    mask = np.zeros(len(a), bool)
    if strategy == "best":
        # rank questions by how well confidence on them predicts the student's
        # mean confidence elsewhere -- computed on the TRAIN FOLD ONLY
        tr = a.iloc[train_idx]
        g = tr.groupby("attempt_id").conf_valid
        s, c = g.transform("sum"), g.transform("count")
        others = (s - tr.conf_valid) / (c - 1).clip(lower=1)
        tmp = tr.assign(_oth=others).dropna(subset=["_oth"])
        score = (tmp.groupby("question_id")
                    .apply(lambda d: d.conf_valid.corr(d._oth) if len(d) > 20 else np.nan)
                    .dropna().sort_values(ascending=False))
        ranked = list(score.index)
        rank_of = {q: i for i, q in enumerate(ranked)}
        for aid, ii in a.groupby("attempt_id").indices.items():
            qs = a.question_id.values[ii]
            order = sorted(range(len(ii)), key=lambda j: rank_of.get(qs[j], 10**6))
            mask[ii[order[:k]]] = True
        return mask, score
    for aid, ii in a.groupby("attempt_id").indices.items():
        if strategy == "first":
            order = np.argsort(a.pos.values[ii])
            mask[ii[order[:k]]] = True
        else:
            sel = rng.choice(len(ii), min(k, len(ii)), replace=False)
            mask[ii[sel]] = True
    return mask, None


def main():
    a = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    a = a[~a.is_unanswered].reset_index(drop=True)
    a["pos"] = a.groupby("attempt_id").cumcount()
    conf = a.conf_valid.values
    y = np.where(conf <= 2, 0, np.where(conf <= 4, 1, 2))     # Bucketing C
    gmean = np.nanmean(conf)
    cat_codes = np.stack([pd.Categorical(a[c].astype(str)).codes for c in CATS], 1)
    levels = [int(cat_codes[:, i].max()) + 1 for i in range(len(CATS))]
    Xnum = np.nan_to_num(a[NUM_FE].values.astype(float))
    print(f"rows={len(a):,}  attempts={a.attempt_id.nunique():,}  "
          f"students={a.student_uid.nunique():,}")
    print(f"median questions per attempt = {int(a.groupby('attempt_id').size().median())}")

    rng = np.random.RandomState(0)
    rows, best_score = [], None
    for strategy in ["first", "random", "best"]:
        for k in [3, 5, 7]:
            P = np.full((len(a), 3), np.nan)
            calib_all = np.zeros(len(a), bool)
            for tr, te in GroupKFold(5).split(a, y, a.student_uid.values):
                mask, sc = pick_calibration(a, k, strategy, tr, rng)
                if sc is not None: best_score = sc
                calib_all |= mask
                CF = calib_features(a, mask, gmean)
                qd = a.iloc[tr].groupby("question_id").is_correct.mean()
                qc = a.question_id.map(qd).fillna(a.is_correct.values[tr].mean()).values.reshape(-1, 1)
                X = np.hstack([Xnum, CF, qc])
                # train on non-calibration rows only (that is the prediction task)
                tr2 = tr[~mask[tr]]
                te2 = te[~mask[te]]
                if len(tr2) < 100 or len(te2) < 20: continue
                P[te2] = fit_predict(X[tr2], cat_codes[tr2], y[tr2],
                                     X[te2], cat_codes[te2], levels)
            ok = np.isfinite(P).all(1)
            pred = P[ok].argmax(1); yy = y[ok]
            base = max(np.bincount(yy, minlength=3) / len(yy))
            rec = precision_recall_fscore_support(yy, pred, labels=[0, 1, 2], zero_division=0)[1]
            rows.append(dict(strategy=strategy, k=k, n_scored=int(ok.sum()),
                             accuracy=accuracy_score(yy, pred), baseline=base,
                             lift=accuracy_score(yy, pred) - base,
                             balanced_acc=balanced_accuracy_score(yy, pred),
                             f1_macro=f1_score(yy, pred, average="macro", zero_division=0),
                             min_recall=rec.min(),
                             qwk=cohen_kappa_score(yy, pred, weights="quadratic"),
                             auc=roc_auc_score(yy, P[ok], multi_class="ovr")))
            print(f"  {strategy:7s} k={k}  acc={rows[-1]['accuracy']:.4f} "
                  f"bal={rows[-1]['balanced_acc']:.4f} n={int(ok.sum()):,}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "calibration_results.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{'#'*105}\nCALIBRATION-BLOCK RESULTS (scored only on NON-calibration questions)\n{'#'*105}")
    print(R.sort_values("accuracy", ascending=False).round(4).to_string(index=False))
    print("\nReference points measured earlier:")
    print("  full LOO (all ratings collected) = 0.8025 | pure cold (none) = 0.4779 | baseline = 0.5013")
    if best_score is not None:
        print(f"\n{'#'*105}\nMOST INFORMATIVE CALIBRATION QUESTIONS\n{'#'*105}")
        print("(correlation between confidence on this question and the student's mean confidence elsewhere)")
        print(best_score.head(12).round(3).to_string())
    print("\nSaved -> calibration_results.csv")


if __name__ == "__main__":
    main()
