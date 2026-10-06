# train_82pct/ — reproduce the 82.34% model

Self-contained. Exactly the four scripts that were run to produce the headline
research result, nothing else.

> **82.34% accuracy · 0.8443 balanced accuracy · 0.7834 macro-F1 · 0.7342 min-class recall · 0.7299 QWK · 0.9516 AUC**
> against a 50.13% majority-class baseline (**+32.2 points**), 5-fold GroupKFold grouped by student.

This is the *research* model (all confidence ratings collected). The model the
web app actually ships is different and scores ~66–72% — see
`../../scripts/fit_web_model.py`. Read `../METHODOLOGY.md` for why they differ.

---

## Contents

| File | Role |
|---|---|
| `idea1_hierarchical.py` | Provides `load_pooled()` and `build_features()`. Not run directly — imported by step 1 |
| `clean_and_eda.py` | **Step 1** — reads the 10 workbooks, applies the 5 integrity fixes, writes `cleaned.pkl` |
| `push_to_90.py` | Provides `rich_loo()` (the 19-feature leave-one-out block), `BASE_FE`, `LOO_FE`. Imported by step 2 |
| `stage6_catemb_search.py` | **Step 2** — CatEmbNet + hyperparameter search. Produces the 82.34% figure |
| `compiled_files` | Symlink to the source data (10 `.xlsx` + 9 embedding `.csv`) |
| `expected_results/` | The CSVs from the original run, to diff against |

**Filenames are load-bearing.** `stage6_catemb_search.py` does
`from push_to_90 import rich_loo, BASE_FE, LOO_FE`, and `clean_and_eda.py` does
`from idea1_hierarchical import load_pooled, build_features`. Renaming breaks
the chain.

---

## Run it

```bash
source /Users/primivista/opt/anaconda3/etc/profile.d/conda.sh
conda activate badminton
cd BIS-quiz/research/train_82pct

python clean_and_eda.py                                    # ~2 min  -> cleaned.pkl
python stage6_catemb_search.py --mode grid --topk 8         # ~2h 40m -> the 82.34%
```

Faster alternatives for step 2:

```bash
python stage6_catemb_search.py --time-one                   # ~10 s, just times one config
python stage6_catemb_search.py --mode random --n 300 --topk 8   # ~28 min, finds the same region
python stage6_catemb_search.py --mode random --n 48             # ~5 min, sanity check
```

Timings measured on this machine (CPU only, no GPU): **5.8 s per config at
3-fold**, 9.7 s at 5-fold. The full grid is 1,728 configs.

### Expected output of step 1

```
[raw] behaviour rows=22,975 attempts=2,646 students=1,950 questions=126
[raw] bank questions=107  embeddings=92
[final] rows=18,980 (from 22,975) | attempts=2,142 | students=1,942 | questions=107
```

Step 2 then drops to **16,990 answered+rated rows** and reports
`features=36 (+1 q_loo_difficulty per fold)`.

### Outputs

| File | Contents |
|---|---|
| `cleaned.pkl` | ~124 MB. Regenerate rather than copy; not committed |
| `cleaned_attempts.pkl` | Attempt-level rollup |
| `eda_report.txt` | The full EDA that drove every modelling decision |
| `stage6_search3.csv` | All 1,728 configs at 3-fold |
| `stage6_final5.csv` | Top 8 re-scored at full 5-fold — **the headline row is here** |

Diff `stage6_final5.csv` against `expected_results/stage6_final5.csv` to confirm
a faithful reproduction.

---

## The winning configuration

```
adam · emb_dim=8 · hidden=(128,64) · lr=0.005 · dropout=0.4 · weight_decay=1e-4 · batch_size=256 · 60 epochs
```

Architecture:

```
36 numeric ─────────────────────────────────┐
q_type       → Embedding(2,  8) ─┐          │
q_blooms     → Embedding(7,  8) ─┼─ 32d ────┴─→ Linear(69→128) → ReLU → BN → Drop(0.4)
q_difficulty → Embedding(5,  8) ─┤               → Linear(128→64) → ReLU → BN → Drop(0.4)
__src        → Embedding(10, 8) ─┘               → Linear(64→3)
```

Target is **Bucketing C**: `1-2 → Low`, `3-4 → Med`, `5 → High`
(9% / 41% / 50%, baseline 50.13%). Loss is class-weighted cross-entropy.

Two features are dropped as statistically zero: `q_feat_parse_tree_depth`
(ρ = −0.007) and `att_fast_frac` (ρ = −0.006).

---

## Two things that make the number honest

**Search selects on balanced accuracy, not accuracy.** The recurring failure on
this data is a class quietly collapsing to ~0 recall while overall accuracy
still looks fine. Ranking on balanced accuracy prevents the search from
optimising straight into that.

**Two-phase protocol.** All 1,728 configs are screened at 3-fold; only the top 8
are re-scored at full 5-fold. Every number quoted elsewhere in this project uses
5-fold, so the finalists are directly comparable.

---

## What carries the result

Essentially all of it is the leave-one-out block in `push_to_90.py::rich_loo()`:

| Stage | Accuracy |
|---|---|
| Direct 3-class training, no LOO | 45.0% |
| **+ leave-one-out features** | **80.6%** |
| + rich 19-feature LOO block | 82.0% |
| + categorical embeddings | 82.2% |
| + this grid search | **82.3%** |

Mean \|Spearman\| is **0.445** for the 19 LOO features versus **0.075** for all
17 behaviour/question features combined. No architecture change or tuning effort
in this project contributed more than 1.5 points.

**Scope condition:** these features need the student's other ratings. With none
collected the model scores **47.8%** — *below* the 50.13% baseline, with per-quiz
AUC of 0.464 / 0.493 / 0.567, i.e. worse than random. That is why the deployed
app collects a 3-question calibration block.

---

## Note on `rich_loo()`

Use the one in `push_to_90.py`. An earlier implementation computed `min`, `max`,
`std` and `nunique` with `groupby.transform()`, which **includes the row
itself** — only the mean was true leave-one-out. That inflated results by ~11
points. The buggy version is kept, deliberately non-importable, at
`../legacy/conf3_pushed_LEAKY.py.txt`. Details in `../METHODOLOGY.md` §11.
