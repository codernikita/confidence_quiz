#!/usr/bin/env python
"""
Direct 3-class confidence model on the T2 (engaged) population — Bucketing B.
=============================================================================

Target (Bucketing B, the only 3-way split measured to carry real skill):
    Low  = confidence 1-3      Med = confidence 4      High = confidence 5

Bucketing A (1-2 / 3 / 4-5) is deliberately NOT used: measured at 70.6% accuracy
against a 71.2% majority baseline, i.e. worse than always guessing "High".

Three models, same folds, same rows:

  M0  BASELINE     majority class / class-prior. The number every result must beat.

  M1  HIER-ORD3    Direct 3-outcome cumulative-link (ordinal) model with crossed
                   random intercepts. K=3 => TWO cutpoints, placed by the optimiser
                   on the Low|Med and Med|High boundaries themselves, rather than
                   inherited from a 5-rung fit and collapsed afterwards.
                     alpha_s ~ N(mu + w*cgpa_s, sigma_a^2)     student effect
                     beta_q  ~ N(W*PCA8(emb_q) + V*meta_q, sigma_b^2)  item effect
                   Unseen students/questions are MARGINALISED over their prior
                   (Monte-Carlo), never plugged in at z=0 -- the link is nonlinear,
                   so the mean of a random effect is not the mean of the prediction.

  M2  HYBRID       "Essence of M1, flexibility of a booster." M1 is fit on the
                   training fold, then its learned structure is distilled into
                   features -- the fitted student effect, item effect, and linear
                   predictor eta -- and handed to an ORDINAL tree ensemble:
                   two monotone binary boosters, P(y>Low) and P(y>High-1),
                   recombined into 3 class probabilities.
                   This keeps the partial pooling that the raw features cannot
                   express, while letting trees find interactions the linear
                   predictor cannot.

Leakage discipline: PCA, LOO-difficulty, scalers, AND the M1 random effects that
feed M2 are all fit on the training fold only. Unseen levels fall back to priors.

Usage:
    python idea1_3class.py                       # T2, cold-attempt + cold-student
    python idea1_3class.py --tier T1 --seeds 0 1 2
    python idea1_3class.py --quick
"""

import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, cohen_kappa_score)

from idea1_hierarchical import load_pooled, build_features, apply_tier_filter

CLASSES = ["Low(1-3)", "Med(4)", "High(5)"]

FIXED = ["opt_sim_chosen_correct", "opt_set_cohesion", "opt_nearest_trap",
         "option_changes", "review_click_count", "marked_for_review",
         "time_rel_student", "time_z_within_q", "is_correct", "q_loo_difficulty"]
QMETA = ["q_flesch_ease", "q_gunning_fog_index", "q_feat_n_words",
         "q_feat_parse_tree_depth", "q_feat_avg_dependency_length"]


def bucket_B(conf):
    """1-3 -> Low(0), 4 -> Med(1), 5 -> High(2)."""
    c = np.clip(conf, 1, 5)
    return np.where(c <= 3, 0, np.where(c == 4, 1, 2))


# ---------------------------------------------------------------------------
# M1 — direct 3-outcome hierarchical ordinal model
# ---------------------------------------------------------------------------

class HierOrdinal3(nn.Module):
    def __init__(self, n_students, n_questions, n_fixed, n_qprior, K=3):
        super().__init__()
        self.K = K
        self.z_s_mu = nn.Parameter(torch.zeros(n_students))
        self.z_s_logs = nn.Parameter(torch.full((n_students,), -2.0))
        self.z_q_mu = nn.Parameter(torch.zeros(n_questions))
        self.z_q_logs = nn.Parameter(torch.full((n_questions,), -2.0))
        self.log_sigma_a = nn.Parameter(torch.tensor(-0.7))
        self.log_sigma_b = nn.Parameter(torch.tensor(-0.7))
        self.mu_a = nn.Parameter(torch.zeros(1))
        self.w_cgpa = nn.Parameter(torch.zeros(1))
        self.W_qprior = nn.Linear(n_qprior, 1)
        self.gamma = nn.Linear(n_fixed, 1, bias=False)
        # K=3 -> exactly two ordered cutpoints, fit directly on Low|Med and Med|High
        self.kappa_1 = nn.Parameter(torch.tensor([-0.5]))
        self.kappa_d = nn.Parameter(torch.zeros(K - 2))

    def cutpoints(self):
        return torch.cat([self.kappa_1,
                          self.kappa_1 + torch.cumsum(F.softplus(self.kappa_d), 0)])

    def alpha(self, s, cgpa, sample=True, z=None):
        zz = self.z_s_mu[s] if z is None else z
        if sample and self.training and z is None:
            zz = zz + torch.randn_like(zz) * self.z_s_logs[s].exp()
        return self.mu_a + self.w_cgpa * cgpa + self.log_sigma_a.exp() * zz

    def beta(self, q, qprior, sample=True, z=None):
        zz = self.z_q_mu[q] if z is None else z
        if sample and self.training and z is None:
            zz = zz + torch.randn_like(zz) * self.z_q_logs[q].exp()
        return self.W_qprior(qprior).squeeze(-1) + self.log_sigma_b.exp() * zz

    def eta(self, s, q, cgpa, qprior, x, sample=True, zs=None, zq=None):
        return (self.alpha(s, cgpa, sample, zs) + self.beta(q, qprior, sample, zq)
                + self.gamma(x).squeeze(-1))

    def logprob(self, eta):
        k = self.cutpoints().unsqueeze(0)
        cdf = torch.sigmoid(k - eta.unsqueeze(1))
        one = torch.ones_like(cdf[:, :1]); zero = torch.zeros_like(cdf[:, :1])
        full = torch.cat([zero, cdf, one], dim=1)
        return (full[:, 1:] - full[:, :-1]).clamp_min(1e-9).log()

    def kl(self):
        def _k(mu, logs):
            s2 = (2 * logs).exp()
            return 0.5 * (s2 + mu ** 2 - 1 - 2 * logs).sum()
        return _k(self.z_s_mu, self.z_s_logs) + _k(self.z_q_mu, self.z_q_logs)

    @torch.no_grad()
    def predict(self, s, q, cgpa, qprior, x, seen_s, seen_q, n_mc=64):
        """Posterior-predictive class probs; unseen levels marginalised over prior."""
        acc = None
        for _ in range(n_mc):
            zs = torch.where(seen_s, self.z_s_mu[s], torch.randn_like(cgpa))
            zq = torch.where(seen_q, self.z_q_mu[q], torch.randn_like(cgpa))
            p = self.logprob(self.eta(s, q, cgpa, qprior, x, False, zs, zq)).exp()
            acc = p if acc is None else acc + p
        p = acc / n_mc
        return p / p.sum(1, keepdim=True)


def prepare_fold(d, emb_cols, tr, n_pca=8):
    d = d.copy()
    dtr = d.iloc[tr]
    d["q_loo_difficulty"] = d.question_id.map(
        dtr.groupby("question_id").is_correct.mean()).fillna(dtr.is_correct.mean())

    has = d._has_emb.values
    Efull = np.nan_to_num(d[emb_cols].values.astype(np.float64))
    tre = [i for i in tr if has[i]]
    ncomp = min(n_pca, max(1, len(set(d.question_id.values[tre])) - 1))
    pca = PCA(ncomp, random_state=0).fit(Efull[tre])
    P = np.zeros((len(d), ncomp)); P[has] = pca.transform(Efull[has])

    meta = d[QMETA].values.astype(np.float64)
    mu, sd = np.nanmean(meta[tr], 0), np.nanstd(meta[tr], 0) + 1e-9
    meta = np.nan_to_num((meta - mu) / sd)

    X = d[FIXED].values.astype(np.float64)
    xmu, xsd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
    X = np.nan_to_num((X - xmu) / xsd)

    cg = d.student_cgpa.values.astype(np.float64)
    cg = np.nan_to_num((cg - np.nanmean(cg[tr])) / (np.nanstd(cg[tr]) + 1e-9))
    return X, np.hstack([P, meta, np.ones((len(d), 1))]), cg


def fit_m1(d, emb_cols, tr, te, y, epochs=1200, lr=0.05, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    X, QP, cg = prepare_fold(d, emb_cols, tr)
    s = pd.Categorical(d.student_uid).codes.astype(np.int64)
    q = pd.Categorical(d.question_id).codes.astype(np.int64)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    Xt, QPt, cgt = T(X), T(QP), T(cg)
    st, qt, yt = T(s, torch.long), T(q, torch.long), T(y, torch.long)

    m = HierOrdinal3(s.max() + 1, q.max() + 1, X.shape[1], QP.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr_t = T(tr, torch.long)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        lp = m.logprob(m.eta(st[tr_t], qt[tr_t], cgt[tr_t], QPt[tr_t], Xt[tr_t]))
        loss = -lp.gather(1, yt[tr_t].unsqueeze(1)).mean() + m.kl() / len(tr)
        loss.backward(); opt.step()

    m.eval()
    seen_s = torch.zeros(s.max() + 1, dtype=torch.bool); seen_s[st[tr_t].unique()] = True
    seen_q = torch.zeros(q.max() + 1, dtype=torch.bool); seen_q[qt[tr_t].unique()] = True
    with torch.no_grad():
        probs_te = m.predict(st[T(te, torch.long)], qt[T(te, torch.long)],
                             cgt[T(te, torch.long)], QPt[T(te, torch.long)],
                             Xt[T(te, torch.long)], seen_s[st[T(te, torch.long)]],
                             seen_q[qt[T(te, torch.long)]]).numpy()
        # ---- distilled "essence" features for M2 (all rows; unseen -> prior) ----
        zs_all = torch.where(seen_s[st], m.z_s_mu[st], torch.zeros_like(cgt))
        zq_all = torch.where(seen_q[qt], m.z_q_mu[qt], torch.zeros_like(cgt))
        alpha = m.alpha(st, cgt, False, zs_all).numpy()
        beta = m.beta(qt, QPt, False, zq_all).numpy()
        eta = m.eta(st, qt, cgt, QPt, Xt, False, zs_all, zq_all).numpy()
        sig = (m.log_sigma_a.exp().item(), m.log_sigma_b.exp().item())
    return probs_te, np.c_[alpha, beta, eta], sig, (X, QP, cg)


def fit_m2(d, X, essence, tr, te, y):
    """Ordinal tree ensemble over raw features + M1's distilled structure.
    Two binary boosters: P(y>0) and P(y>1); recombined and monotonised."""
    Z = np.hstack([X, essence])
    p_gt = []
    for thr in [0, 1]:
        b = (y > thr).astype(int)
        mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, verbose=-1, n_jobs=8)
        mdl.fit(Z[tr], b[tr])
        p_gt.append(mdl.predict_proba(Z[te])[:, 1])
    p1, p2 = p_gt[0], np.minimum(p_gt[0], p_gt[1])          # enforce monotonicity
    P = np.c_[1 - p1, p1 - p2, p2].clip(1e-9)
    return P / P.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def report(y, P, tag, prior_correct=True):
    out = {"model": tag}
    yp = P.argmax(1)
    rows = {}
    rules = {"argmax": yp}
    if prior_correct:
        rules["prior-corrected"] = (P / P.mean(0, keepdims=True).clip(1e-9)).argmax(1)
    for rn, pred in rules.items():
        acc = accuracy_score(y, pred)
        bal = balanced_accuracy_score(y, pred)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
        key = "" if rn == "argmax" else "_bal"
        out.update({f"acc{key}": acc, f"balacc{key}": bal,
                    f"prec_macro{key}": pm, f"rec_macro{key}": rm, f"f1_macro{key}": fm,
                    f"prec_wtd{key}": pw, f"rec_wtd{key}": rw, f"f1_wtd{key}": fw,
                    f"qwk{key}": cohen_kappa_score(y, pred, weights="quadratic")})
        rows[rn] = pd.DataFrame({"class": CLASSES, "precision": p, "recall": r,
                                 "f1": f, "support": s})
    try:
        out["auc_ovr"] = roc_auc_score(y, P, multi_class="ovr")
    except Exception:
        out["auc_ovr"] = np.nan
    return out, rows, confusion_matrix(y, yp, labels=[0, 1, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="T2", choices=["none", "T1", "T2", "T3"])
    ap.add_argument("--splits", nargs="+", default=["attempt", "student"],
                    choices=["attempt", "student", "question", "random"])
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="idea1_3class_results")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.splits, a.seeds = 200, ["attempt"], [0]

    print("=" * 78)
    print("DIRECT 3-CLASS CONFIDENCE MODEL — Bucketing B, tier", a.tier)
    print("=" * 78)
    B, Q, E = load_pooled(verbose=False)
    d, emb_cols = build_features(B, Q, E, verbose=False)
    d = apply_tier_filter(d, a.tier)
    d = d[d.answered == 1].reset_index(drop=True)
    y = bucket_B(d.confidence_rating.values)
    print(f"  answered rows: {len(d):,} | students {d.student_uid.nunique():,} | "
          f"questions {d.question_id.nunique()}")
    print(f"  class balance: " + "  ".join(
        f"{c}={np.mean(y == i):.1%}" for i, c in enumerate(CLASSES)))
    os.makedirs(a.out, exist_ok=True)

    all_rows = []
    for split in a.splits:
        gcol = {"attempt": "attempt_id", "student": "student_uid",
                "question": "question_id"}.get(split)
        for seed in a.seeds:
            splitter = (KFold(5, shuffle=True, random_state=seed) if gcol is None
                        else GroupKFold(5))
            groups = None if gcol is None else d[gcol].values
            oof1 = np.full((len(d), 3), np.nan)
            oof2 = np.full((len(d), 3), np.nan)
            sig_a, sig_b = [], []
            t0 = time.time()
            for k, (tr, te) in enumerate(splitter.split(d, y, groups)):
                P1, ess, sig, (X, QP, cg) = fit_m1(d, emb_cols, tr, te, y,
                                                   epochs=a.epochs, seed=seed)
                oof1[te] = P1
                oof2[te] = fit_m2(d, X, ess, tr, te, y)
                sig_a.append(sig[0]); sig_b.append(sig[1])
                print(f"    [{split} seed{seed}] fold {k+1}/5 ({time.time()-t0:.0f}s) "
                      f"sigma_s={sig[0]:.3f} sigma_q={sig[1]:.3f}")

            print(f"\n{'='*78}\nSPLIT = COLD-{split.upper()}  (seed {seed})\n{'='*78}")
            base_pred = np.full(len(y), np.bincount(y).argmax())
            m0, _, _ = report(y, np.eye(3)[base_pred], "M0 majority", prior_correct=False)
            m1, r1, cm1 = report(y, oof1, "M1 hier-ord3")
            m2, r2, cm2 = report(y, oof2, "M2 hybrid")
            for m in (m0, m1, m2):
                m["split"], m["seed"] = split, seed
                m["sigma_student"] = float(np.mean(sig_a))
                m["sigma_question"] = float(np.mean(sig_b))
            all_rows += [m0, m1, m2]

            cols = ["model", "acc", "balacc", "prec_macro", "rec_macro", "f1_macro",
                    "f1_wtd", "qwk", "auc_ovr"]
            T = pd.DataFrame([m0, m1, m2])
            pd.set_option("display.width", 200)
            print("\n-- argmax rule --")
            print(T[[c for c in cols if c in T.columns]].round(4).to_string(index=False))
            colsb = ["model", "acc_bal", "balacc_bal", "prec_macro_bal",
                     "rec_macro_bal", "f1_macro_bal", "qwk_bal"]
            print("\n-- prior-corrected rule --")
            print(T[[c for c in colsb if c in T.columns]].round(4).to_string(index=False))

            for nm, rr, cm in [("M1", r1, cm1), ("M2", r2, cm2)]:
                print(f"\n  [{nm}] per-class (argmax):")
                print(rr["argmax"].round(4).to_string(index=False))
                print(f"  [{nm}] confusion (rows=true, cols=pred):")
                print(pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                                   columns=[f"pred_{c}" for c in CLASSES]).to_string())
                rr["argmax"].to_csv(os.path.join(a.out, f"{split}_seed{seed}_{nm}_perclass.csv"),
                                    index=False)

    R = pd.DataFrame(all_rows)
    R.to_csv(os.path.join(a.out, "all_metrics.csv"), index=False)
    print(f"\nSaved -> {a.out}/all_metrics.csv")


if __name__ == "__main__":
    main()
