#!/usr/bin/env python
"""
STAGE 5 — everything not yet tried: unused columns, encodings, PCA, deep learning.

The audit found real gaps in what was being fed to the model:
  * 115 varying syn_* columns (POS counts/ratios, dependency counts/ratios,
    parse statistics) sat in Question_Bank and were NEVER merged into the frame.
  * 16 feat_* readability/lexical columns, likewise mostly unmerged.
  * 4 categoricals never encoded at all: q_type, q_blooms_level, q_difficulty,
    and __src (the SUBJECT -- DBMS vs OOPS vs DSA etc.), which plausibly shifts
    how students rate.

Blocks tested, cumulatively and in isolation:
  BEST        = Stage-4 winner (base behaviour + rich leave-one-out)
  +CAT        = one-hot q_type / q_blooms_level / q_difficulty / subject
  +SYN        = the 131 unmerged question columns, PCA-reduced inside each fold
  +ENG        = engineered interactions (student ability x item difficulty, etc.)
  ALL         = everything
  ALL+PCA     = everything, then a global PCA on the whole design matrix

Models: tuned LightGBM (Stage-4 champion) plus three deep/other learners --
an MLP, a categorical-embedding neural net, and a scaled linear model -- to
settle whether deep learning helps on this data or, as in every previous round,
loses to gradient boosting.

Controls unchanged: GroupKFold by student_uid, all encoders/PCA/target-encodings
fit inside the training fold only, min_class_recall reported so a collapsed
class cannot masquerade as a good score.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             roc_auc_score)
import lightgbm as lgb
from idea1_hierarchical import load_pooled
from push_to_90 import rich_loo, LOO_FE, BASE_FE

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); np.random.seed(0)

CATS = ["q_type", "q_blooms_level", "q_difficulty", "__src"]


def build():
    a = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    a = a[~a.is_unanswered].reset_index(drop=True)
    a = rich_loo(a)

    # ---- merge the 131 unused question columns ----
    _, Q, _ = load_pooled(verbose=False)
    num = Q.select_dtypes(include=[np.number])
    synf = [c for c in num.columns if c.startswith(("syn_", "feat_"))
            and num[c].nunique() > 1]
    QN = Q[["question_id"]].join(num[synf])
    a = a.merge(QN, on="question_id", how="left", suffixes=("", "_qq"))
    synf = [c for c in synf if c in a.columns]
    print(f"  merged {len(synf)} previously-unused question columns")

    # ---- engineered interactions ----
    a["eng_cgpa_x_qdiff"] = a.student_cgpa * a.get("q_loo_diff_tmp", 1.0)
    a["eng_time_x_changes"] = a.time_z_within_q * (a.option_changes + 1)
    a["eng_conf_gap"] = a.loo_conf_mean - a.loo_median
    a["eng_spread"] = a.loo_conf_max - a.loo_conf_min
    a["eng_mode_dom"] = a.loo_mode_share * a.loo_n
    a["eng_hesitate"] = (a.option_changes > 0).astype(float) + a.marked_for_review \
        + (a.review_click_count > 0).astype(float)
    a["eng_time_per_word"] = a.time_spent / (a.q_feat_n_words.fillna(15) + 1)
    ENG = ["eng_cgpa_x_qdiff", "eng_time_x_changes", "eng_conf_gap", "eng_spread",
           "eng_mode_dom", "eng_hesitate", "eng_time_per_word"]

    # ---- one-hot categoricals ----
    D = pd.get_dummies(a[CATS].astype(str), prefix=CATS, dummy_na=False)
    CATCOLS = list(D.columns)
    a = pd.concat([a, D.astype(float)], axis=1)
    print(f"  one-hot categoricals -> {len(CATCOLS)} columns")
    return a, synf, ENG, CATCOLS


class CatEmbNet(nn.Module):
    """Neural net with learned embeddings for the categorical blocks."""
    def __init__(self, n_num, n_cat_levels, emb=4, hid=128):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, emb) for n in n_cat_levels])
        d = n_num + emb * len(n_cat_levels)
        self.net = nn.Sequential(
            nn.Linear(d, hid), nn.ReLU(), nn.BatchNorm1d(hid), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.BatchNorm1d(hid // 2),
            nn.Dropout(0.2), nn.Linear(hid // 2, 3))

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, dim=1))


def fit_catemb(Xtr, ctr, ytr, Xte, cte, n_levels, epochs=60):
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ctr_t = torch.tensor(ctr, dtype=torch.long); cte_t = torch.tensor(cte, dtype=torch.long)
    yt = torch.tensor(ytr, dtype=torch.long)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)),
                     dtype=torch.float32)
    net = CatEmbNet(Xtr.shape[1], n_levels)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    n = len(ytr)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            if len(b) < 2: continue
            opt.zero_grad()
            loss = lossf(net(Xtr_t[b], ctr_t[b]), yt[b])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(Xte_t, cte_t), dim=1).numpy()


def report(y, P, block, model):
    pred = P.argmax(1)
    base = max(np.bincount(y, minlength=3) / len(y))
    rec = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)[1]
    acc = accuracy_score(y, pred)
    return dict(block=block, model=model, accuracy=acc, baseline=base, lift=acc - base,
                balanced_acc=balanced_accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                min_recall=rec.min(),
                qwk=cohen_kappa_score(y, pred, weights="quadratic"),
                auc=roc_auc_score(y, P, multi_class="ovr"),
                hits90="YES" if acc >= 0.90 else "-")


def main():
    a, SYN, ENG, CATCOLS = build()
    c = a.conf_valid.values
    y = np.where(c <= 2, 0, np.where(c <= 4, 1, 2))          # Bucketing C
    BEST = BASE_FE + LOO_FE
    BLOCKS = {
        "BEST (stage-4)":  (BEST, False),
        "+CAT":            (BEST + CATCOLS, False),
        "+SYN(pca)":       (BEST, True),
        "+ENG":            (BEST + ENG, False),
        "ALL":             (BEST + CATCOLS + ENG, True),
    }
    cat_codes = np.stack([pd.Categorical(a[c_].astype(str)).codes for c_ in CATS], 1)
    n_levels = [int(cat_codes[:, i].max()) + 1 for i in range(len(CATS))]
    rows = []

    for bname, (cols, use_syn) in BLOCKS.items():
        X0 = np.nan_to_num(a[cols].values.astype(float))
        store = {k: np.full((len(a), 3), np.nan)
                 for k in ["LGBM-tuned", "MLP", "CatEmbNet", "LogReg", "ALL+PCA-LGBM"]}
        for tr, te in GroupKFold(5).split(a, y, a.student_uid.values):
            qd = a.iloc[tr].groupby("question_id").is_correct.mean()
            qc = a.question_id.map(qd).fillna(a.is_correct.values[tr].mean()).values.reshape(-1, 1)
            parts = [X0, qc]
            if use_syn and SYN:
                S = np.nan_to_num(a[SYN].values.astype(float))
                p = PCA(12, random_state=0).fit(S[tr])
                parts.append(p.transform(S))
            X = np.hstack(parts)
            Xtr, Xte, ytr = X[tr], X[te], y[tr]

            m = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.02, num_leaves=63,
                                   min_child_samples=15, subsample=0.8,
                                   colsample_bytree=0.7, reg_lambda=1.0,
                                   class_weight="balanced", verbose=-1, n_jobs=6,
                                   random_state=0).fit(Xtr, ytr)
            store["LGBM-tuned"][te] = m.predict_proba(Xte)

            store["MLP"][te] = make_pipeline(StandardScaler(),
                MLPClassifier((128, 64), max_iter=350, random_state=0,
                              early_stopping=True)).fit(Xtr, ytr).predict_proba(Xte)
            store["LogReg"][te] = make_pipeline(StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced")
                ).fit(Xtr, ytr).predict_proba(Xte)
            store["CatEmbNet"][te] = fit_catemb(Xtr, cat_codes[tr], ytr,
                                                Xte, cat_codes[te], n_levels)
            # global PCA over the whole design matrix
            sc = StandardScaler().fit(Xtr)
            pg = PCA(0.95, random_state=0).fit(sc.transform(Xtr))
            store["ALL+PCA-LGBM"][te] = lgb.LGBMClassifier(
                n_estimators=800, learning_rate=0.03, num_leaves=31,
                class_weight="balanced", verbose=-1, n_jobs=6, random_state=0
                ).fit(pg.transform(sc.transform(Xtr)), ytr
                      ).predict_proba(pg.transform(sc.transform(Xte)))
        for k, P in store.items():
            rows.append(report(y, P, bname, k))
        print(f"  done block: {bname}")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "stage5_results.csv"), index=False)
    pd.set_option("display.width", 200)
    show = ["block", "model", "accuracy", "baseline", "lift", "balanced_acc",
            "f1_macro", "min_recall", "qwk", "auc", "hits90"]
    print(f"\n{'#'*110}\nALL RESULTS (sorted by accuracy)\n{'#'*110}")
    print(R.sort_values("accuracy", ascending=False)[show].round(4).to_string(index=False))
    print(f"\n{'#'*110}\nBEST PER BLOCK  (did the new features help?)\n{'#'*110}")
    print(R.loc[R.groupby('block').accuracy.idxmax()].sort_values(
        "accuracy", ascending=False)[show].round(4).to_string(index=False))
    print(f"\n{'#'*110}\nBEST PER MODEL  (does deep learning win?)\n{'#'*110}")
    print(R.loc[R.groupby('model').accuracy.idxmax()].sort_values(
        "accuracy", ascending=False)[show].round(4).to_string(index=False))
    print("\nSaved -> stage5_results.csv")


if __name__ == "__main__":
    main()
