#!/usr/bin/env python
"""
STAGE 6 — CatEmbNet hyperparameter search.

Changes vs Stage 5:
  * DROPS the two dead features measured at |Spearman| < 0.01 against the target:
        q_feat_parse_tree_depth  (-0.0074)
        att_fast_frac            (-0.0060)
  * Searches optimizer / embedding width / hidden architecture / learning rate /
    dropout / weight decay / batch size.

Two-phase protocol so the search is affordable AND the winner is honestly scored:
  PHASE 1  every candidate scored with 3-fold GroupKFold (grouped by student).
  PHASE 2  the top-K candidates re-scored with the full 5-fold GroupKFold,
           identical to every previously reported number, so the result is
           directly comparable to the 82.22% baseline.

Selection metric is BALANCED ACCURACY, not raw accuracy: the recurring failure
mode in this project has been a class collapsing while overall accuracy looks
fine. Raw accuracy, macro-F1, min-class recall, QWK and AUC are all logged too.

Usage:
    python stage6_catemb_search.py                 # random search, 48 configs
    python stage6_catemb_search.py --n 24          # quicker
    python stage6_catemb_search.py --mode grid     # full structured grid
    python stage6_catemb_search.py --time-one      # time a single config, then exit
"""
import argparse, itertools, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             roc_auc_score)
from push_to_90 import rich_loo, BASE_FE, LOO_FE

OUT = os.path.dirname(os.path.abspath(__file__))
CATS = ["q_type", "q_blooms_level", "q_difficulty", "__src"]
DEAD = ["q_feat_parse_tree_depth", "att_fast_frac"]      # |rho| < 0.01 -> dropped

SPACE = {
    "optimizer":  ["adamw", "adam", "sgd", "rmsprop"],
    "emb_dim":    [2, 4, 8],
    "hidden":     [(64, 32), (128, 64), (256, 128)],
    "lr":         [5e-4, 1e-3, 2e-3, 5e-3],
    "dropout":    [0.2, 0.3, 0.4],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [256, 512],
}


class CatEmbNet(nn.Module):
    def __init__(self, n_num, levels, emb_dim, hidden, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in levels])
        d = n_num + emb_dim * len(levels)
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, dim=1))


def make_opt(name, params, lr, wd):
    if name == "adamw":   return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":    return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)


def fit_one(Xtr, ctr, ytr, Xte, cte, levels, cfg, epochs=60, seed=0):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ct = torch.tensor(ctr, dtype=torch.long); ce = torch.tensor(cte, dtype=torch.long)
    yt = torch.tensor(ytr, dtype=torch.long)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)),
                     dtype=torch.float32)
    net = CatEmbNet(Xtr.shape[1], levels, cfg["emb_dim"], cfg["hidden"], cfg["dropout"])
    opt = make_opt(cfg["optimizer"], net.parameters(), cfg["lr"], cfg["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=w)
    bs, n = cfg["batch_size"], len(ytr)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2: continue
            opt.zero_grad(); lossf(net(xt[b], ct[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(xe, ce), dim=1).numpy()


def cv_score(a, X, cat_codes, y, levels, cfg, n_splits, epochs=60):
    P = np.full((len(a), 3), np.nan)
    for tr, te in GroupKFold(n_splits).split(a, y, a.student_uid.values):
        qd = a.iloc[tr].groupby("question_id").is_correct.mean()
        qc = a.question_id.map(qd).fillna(a.is_correct.values[tr].mean()).values.reshape(-1, 1)
        Xf = np.hstack([X, qc])
        P[te] = fit_one(Xf[tr], cat_codes[tr], y[tr], Xf[te], cat_codes[te], levels, cfg, epochs)
    pred = P.argmax(1)
    rec = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)[1]
    return dict(accuracy=accuracy_score(y, pred),
                balanced_acc=balanced_accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                min_recall=rec.min(),
                qwk=cohen_kappa_score(y, pred, weights="quadratic"),
                auc=roc_auc_score(y, P, multi_class="ovr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "grid"])
    ap.add_argument("--n", type=int, default=48, help="configs for random search")
    ap.add_argument("--topk", type=int, default=5, help="finalists re-run at 5-fold")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-one", action="store_true")
    a_ = ap.parse_args()
    rng = np.random.RandomState(a_.seed)

    a = pd.read_pickle(os.path.join(OUT, "cleaned.pkl"))
    a = a[~a.is_unanswered].reset_index(drop=True)
    a = rich_loo(a)
    feats = [f for f in (BASE_FE + LOO_FE) if f not in DEAD]
    print(f"rows={len(a):,}  students={a.student_uid.nunique():,}")
    print(f"features={len(feats)} (+1 q_loo_difficulty per fold)  DROPPED: {DEAD}")

    X = np.nan_to_num(a[feats].values.astype(float))
    cat_codes = np.stack([pd.Categorical(a[c].astype(str)).codes for c in CATS], 1)
    levels = [int(cat_codes[:, i].max()) + 1 for i in range(len(CATS))]
    c = a.conf_valid.values
    y = np.where(c <= 2, 0, np.where(c <= 4, 1, 2))

    if a_.time_one:
        cfg = dict(optimizer="adamw", emb_dim=4, hidden=(128, 64), lr=2e-3,
                   dropout=0.3, weight_decay=1e-4, batch_size=512)
        t0 = time.time(); cv_score(a, X, cat_codes, y, levels, cfg, 3, a_.epochs)
        per3 = time.time() - t0
        print(f"\nONE CONFIG @ 3-fold = {per3:.1f}s   (5-fold approx {per3*5/3:.1f}s)")
        print(f"  -> {a_.n} configs search  ~= {per3*a_.n/60:.1f} min")
        print(f"  -> + {a_.topk} finalists @5-fold ~= {per3*5/3*a_.topk/60:.1f} min")
        print(f"  -> TOTAL ESTIMATE ~= {(per3*a_.n + per3*5/3*a_.topk)/60:.1f} min")
        return

    if a_.mode == "grid":
        keys = list(SPACE)
        cands = [dict(zip(keys, v)) for v in itertools.product(*SPACE.values())]
        print(f"FULL GRID = {len(cands)} configs")
    else:
        cands = []
        seen = set()
        while len(cands) < a_.n:
            cfg = {k: SPACE[k][rng.randint(len(SPACE[k]))] for k in SPACE}
            key = json.dumps({k: str(v) for k, v in cfg.items()}, sort_keys=True)
            if key in seen: continue
            seen.add(key); cands.append(cfg)
        print(f"RANDOM SEARCH = {len(cands)} configs")

    rows, t0 = [], time.time()
    for i, cfg in enumerate(cands, 1):
        s = cv_score(a, X, cat_codes, y, levels, cfg, 3, a_.epochs)
        rows.append({**{k: str(v) for k, v in cfg.items()}, **s, "phase": "search3"})
        el = time.time() - t0
        print(f"  [{i:3d}/{len(cands)}] {cfg['optimizer']:7s} emb={cfg['emb_dim']} "
              f"h={str(cfg['hidden']):10s} lr={cfg['lr']:.0e} do={cfg['dropout']} "
              f"wd={cfg['weight_decay']:.0e} bs={cfg['batch_size']} | "
              f"acc={s['accuracy']:.4f} bal={s['balanced_acc']:.4f} "
              f"| {el/60:.1f}m elapsed, ETA {el/i*(len(cands)-i)/60:.1f}m", flush=True)

    R = pd.DataFrame(rows).sort_values("balanced_acc", ascending=False)
    R.to_csv(os.path.join(OUT, "stage6_search3.csv"), index=False)
    print(f"\n{'#'*110}\nTOP 15 @ 3-FOLD (ranked by balanced accuracy)\n{'#'*110}")
    print(R.head(15).round(4).to_string(index=False))

    print(f"\n{'#'*110}\nPHASE 2 — TOP {a_.topk} RE-SCORED AT FULL 5-FOLD\n{'#'*110}")
    fin = []
    for i in range(min(a_.topk, len(R))):
        r = R.iloc[i]
        cfg = dict(optimizer=r.optimizer, emb_dim=int(r.emb_dim),
                   hidden=eval(r.hidden), lr=float(r.lr), dropout=float(r.dropout),
                   weight_decay=float(r.weight_decay), batch_size=int(r.batch_size))
        s = cv_score(a, X, cat_codes, y, levels, cfg, 5, a_.epochs)
        fin.append({**{k: str(v) for k, v in cfg.items()}, **s, "phase": "final5"})
        print(f"  finalist {i+1}: acc={s['accuracy']:.4f} bal={s['balanced_acc']:.4f} "
              f"f1={s['f1_macro']:.4f} minrec={s['min_recall']:.4f} "
              f"qwk={s['qwk']:.4f} auc={s['auc']:.4f}", flush=True)
    F = pd.DataFrame(fin).sort_values("balanced_acc", ascending=False)
    F.to_csv(os.path.join(OUT, "stage6_final5.csv"), index=False)
    print(f"\n{'#'*110}\nFINAL RANKING (5-fold, comparable to the 82.22% baseline)\n{'#'*110}")
    print(F.round(4).to_string(index=False))
    b = F.iloc[0]
    print(f"\nBEST: acc={b.accuracy:.4f}  balanced_acc={b.balanced_acc:.4f}  "
          f"min_recall={b.min_recall:.4f}  qwk={b.qwk:.4f}  auc={b.auc:.4f}")
    print("Baseline to beat (Stage-5 CatEmbNet): acc=0.8222  balanced_acc=0.8217")
    print("\nSaved -> stage6_search3.csv, stage6_final5.csv")


if __name__ == "__main__":
    main()
