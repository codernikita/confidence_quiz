# research/ — model development archive

The offline research that produced the model shipped in `../src/model/`.
Nothing here runs in the web app. It exists so the numbers quoted in
`CONTEXT.md`, `../README.md` and the report are reproducible.

**The production path is elsewhere:**
`../scripts/fit_web_model.py` → `../src/model/coefficients.json` → `../src/model/predict.js`

This folder is the *evidence*; that path is the *deliverable*.

---

## ⚠️ Two traps before you touch anything

**1. `legacy/conf3_pushed_LEAKY.py.txt` — do not run or import.**
Its `add_loo_features()` computes `min`/`max`/`std`/`nunique` with
`groupby.transform()`, which **includes the row itself**. Only `mean` was true
leave-one-out. It differed from correct LOO on 3.8% / 1.3% / 6.9% of rows and
**inflated accuracy by ~11 points** (reported 80.6% vs the honest 82.0% from a
richer, clean feature set). Kept only as a record of the bug. Deliberately
renamed `.py.txt` so `import` cannot reach it.

The correct implementation is **`push_to_90.py::rich_loo()`**. Use that one.

**2. Never predict `is_correct` using `opt_sim_chosen_correct`.**
That feature is `cosine(chosen_option, correct_answer)` — it equals exactly 1.0
for **97% of correct answers**, i.e. it *is* the label. It produced a fake 98.8%
accuracy / 0.999 AUC before being caught. Same for any "hesitated" target built
from `option_changes` / `marked_for_review` / `review_click_count` while those
are in the feature set — a tautology that scored a fake 100.0%.

---

## Prerequisites

```bash
source /Users/primivista/opt/anaconda3/etc/profile.d/conda.sh
conda activate badminton
```

Needs `../../cleaned.pkl` (~124 MB), produced by step 1. Not committed —
regenerate it rather than copying it around.

---

## Run order

| # | Script | Produces | Purpose |
|---|---|---|---|
| 1 | `clean_and_eda.py` | `cleaned.pkl`, `eda_report.txt` | **Run first.** 22,975 → 16,990 rows; all five integrity fixes; the EDA that drove every later decision |
| 2 | `push_to_90.py` | `push_to_90_results.csv` | **Defines `rich_loo()`** — the correct 19-feature LOO block. Steps 3 and 4 import from here, so do not rename or refactor it casually |
| 3 | `stage5_allfeatures.py` | `stage5_results.csv` | Ablation: 131 unused `syn_*` columns, one-hot vs learned categorical embeddings, engineered interactions, global PCA, MLP vs CatEmbNet |
| 4 | `stage6_catemb_search.py` | `stage6_search3.csv`, `stage6_final5.csv` | Exhaustive 1,728-config CatEmbNet grid → the **82.34%** headline |
| 5 | `calibration_sweep.py` | `calibration_sweep.csv` | Per-quiz k-sweep with randomised calibration assignment → **k=3 is the knee** |
| 6 | `calib_difficulty_test.py` | stdout | easy/med/hard vs random vs medium-only vs easiest-only |
| 7 | `cross_domain_test.py` | stdout | Does confidence style transfer between subjects? (It partly does, and it isn't enough) |
| 8 | `model_bakeoff.py` | `bakeoff_results.csv` | Target × model grid — found the 91.5% attempt-level pass/fail result |

Supporting / superseded:

| Script | Note |
|---|---|
| `idea1_hierarchical.py` | Hierarchical Bayesian ordinal + careless mixture. Reached 45% on the 5-class target. Also contains the `load_pooled` / `build_features` helpers others import |
| `idea1_3class.py` | Direct 3-class hierarchical + a hybrid that distils random effects into booster features |
| `calibration_block.py` | **Superseded by `calibration_sweep.py`.** Its "first-k" selection is confounded with question identity, because question order is fixed per workbook (1–4 distinct orders exist). Kept for the audit trail |

Steps 5–8 need only `cleaned.pkl`; they do not depend on 2–4.

---

## Headline results

| Setting | Accuracy | Baseline | Notes |
|---|---|---|---|
| Full LOO, all ratings collected | **82.34%** | 50.13% | Research ceiling. CatEmbNet, bal-acc 0.844, AUC 0.952 |
| **k=3 calibration** (what ships) | **~66–72%** | ~48% | Varies by quiz. AUC 0.76–0.83 |
| No ratings at all (k=0) | **47.8%** | 50.13% | **Below baseline.** Per-quiz AUC 0.464 / 0.493 / 0.567 — *anti-predictive* |

**Why the calibration block exists:** confidence prediction from behaviour alone
does not work. That was tested across four architectures, six model families,
three bucketings, three engagement tiers and a 1,728-config grid. Two of three
quizzes rank *worse than random* at k=0. Collecting 3 ratings is what makes the
feature viable at all.

---

## Validation protocol (non-negotiable)

- **GroupKFold by `student_uid`.** Each student has ~9.4 rows (max 33) and their
  own mean predicts their individual ratings at ρ = 0.719. Random K-fold reported
  ρ = 0.533 where student-grouped reported ρ = 0.252 — a **2× overstatement**.
- Every fold-dependent transform (`q_loo_difficulty`, PCA, scalers) is fit on the
  **training fold only**.
- Always print accuracy **next to its majority-class baseline** *and*
  **min-class recall**. The recurring failure here is a class collapsing to ~0
  recall while overall accuracy still looks fine.
- Calibration experiments score **only non-calibration rows**, and average over
  8 random assignments so no individual question can drive the result.

---

## What was tried and rejected

| Approach | Measured effect |
|---|---|
| 768-d SBERT embeddings | **−0.16 acc.** ρ 0.225 random-split → 0.049 question-grouped = ~78% memorisation |
| 131 unused `syn_*` / `feat_*` question columns | +0.21 — noise |
| One-hot categoricals into LightGBM | −0.23 (the *same* fields as learned embeddings: **+0.45**) |
| 7 engineered interaction features | −0.13 |
| Global PCA over the design matrix | **−10.3** — blends the LOO signal into noise |
| Ordinal decomposition (2 binary + monotone) | −0.9 |
| CatBoost | −1.5 (best min-recall, 0.766) |
| Soft-voting ensemble | +0.3 |
| Plain MLP, no categorical embeddings | 77.7% vs 82.2% — **embeddings are the entire gap** |
| Bucketing A (1-2 / 3 / 4-5) | 70.6% acc against a **71.2% baseline** — zero skill |
| Cross-subject calibration | AUC 0.680 vs 0.830 same-subject; 9 foreign items < 3 local ones |
| Two-tower net (SBERT + option attention) | 40% — lost head-to-head to the hierarchical model |

Dropped as statistically zero: `q_feat_parse_tree_depth` (ρ = −0.007),
`att_fast_frac` (ρ = −0.006).

---

## Open items

- Extend the learning-rate sweep **above 0.005** — the grid saturated at its
  upper bound (17 of the top 20 configs picked the maximum value, and 19 of 20
  picked Adam over AdamW).
- Consider grid finalist **#2** (`emb=2, (128,64), lr=5e-3, do=0.3, wd=1e-3, bs=512`):
  it wins on 5 of 6 metrics including min-class recall (0.771 vs 0.734), losing
  only balanced accuracy by 0.4.
- Sensitivity-check the T2 engagement thresholds (≥3 distinct values, <20% blank).
  Those are judgment calls, not derived.
- `calib_difficulty_test.py` did not complete for DBMS — its difficulty terciles
  are too thin at 15 questions to fill three slots under the constraint.
