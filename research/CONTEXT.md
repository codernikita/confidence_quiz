# CONTEXT.md — Student Confidence Prediction Project

> Working memory for this project. Read this first before picking the work back up.
> Every number below was measured on this dataset, not estimated.

---

## 1. What we are building

A model that predicts a student's **confidence** (Low / Medium / High) on each quiz
question from behavioural telemetry, so a results page can show per-question
confidence alongside correctness.

**End goal:** a quiz website (React) that runs the quiz, logs telemetry, and at
submission shows an analysis page — per-question confidence, correctness, and
behavioural state — with seaborn-style plots.

---

## 2. Environment

| Thing | Value |
|---|---|
| Conda env | `badminton` at `/Users/primivista/opt/anaconda3/envs/badminton` |
| Activate | `source /Users/primivista/opt/anaconda3/etc/profile.d/conda.sh && conda activate badminton` |
| Python | 3.10.18 · torch 2.7.1 · lightgbm 4.6.0 · sklearn 1.4.2 · pandas 1.5.3 |
| Project dir | `/Users/primivista/Downloads/Compiled_files` |
| Quiz app | `/Users/primivista/Desktop/Bis-quiz` (React + Firebase, working) |

**Known env issues**
- `transformers` crashes on *any* `from_pretrained` weight load (mutex error).
  Confirmed universal, even on a 4KB test model. Do not rely on live embedding.
- `sentence_transformers` fails to import for the same reason.
- LibreOffice not installed → cannot render docx to PDF for visual checks.
- `docx` npm package installed locally in the project dir.

---

## 3. Data

10 Excel workbooks in `compiled_files/*/`. Each has `ML_Behavioral_Data`
(one row per student-question), `Question_Bank`, `Student_Profiles`.
Also `*_BERT/RoBERTa/SentenceBERT_embeddings.csv` (768-d, question-level, 92 questions).

### Cleaning (`clean_and_eda.py` → `cleaned.pkl`)

Raw **22,975 rows** → clean **16,990 answered rows** / 1,962 attempts / 1,803 students / 107 questions.

| Fix | Removed |
|---|---|
| Questions absent from Question_Bank (incl. orphan 2nd form in `dms_quiz_one`) | 534 rows / 39 attempts |
| **Repeat sittings** — kept only each student's FIRST attempt per subject | 453 attempts / 3,461 rows |
| `confidence==0` conflated "unsure" with "did not answer" → split into flag | 1,990 rows |
| Answered-but-rated-0 → clipped to 1 | 297 rows |
| Duplicate bank cols 155–165; taxonomy casing (Theory/theoretical, Apply/Applying) | 11 cols |

### Per-quiz sizes (cleaned, answered)

| Quiz | rows | attempts | q/attempt |
|---|---|---|---|
| DBMS | 714 | 49 | **15** |
| DMS-2 | 1,838 | 146 | 13 |
| dms_quiz_one | 1,735 | 138 | 13 |
| ELC-1 | 3,024 | 276 | 11 |
| ELC-3 | 2,829 | 291 | 10 |
| ELC-2 | 3,546 | 715 | 5 |

---

## 4. The five findings that shaped everything

1. **53% of confidence variance is BETWEEN STUDENTS**, only 7% between questions.
   Who is rating >> what is being rated. A student's own mean rating predicts
   their individual ratings at Spearman **0.719**.
2. **`confidence==0` is mostly "did not answer"** — 350 of 423 zeros in dms_quiz_one
   were unanswered items with `time_spent = 0`.
3. **"Fast" means opposite things by engagement.** Engaged students at 3–5s are
   **86.7% correct** (mastery); disengaged at 3–5s are **15.4%** (abandonment).
   `corr(total_time, CGPA) = −0.306` — stronger students finish *faster*.
4. **38.5% of attempts are straightlined.** Careless rows are the *easiest* to
   predict (constant target) so they inflate any pooled metric: same model scored
   ρ=0.44 on straightlined rows vs ρ=0.29 on clean rows.
5. **Confusion is separable from misconception** — near-identical CGPA (6.75 vs
   6.86) but 2.4× time difference (104s vs 43s). Time is the discriminator.

---

## 5. Final model

**CatEmbNet** — categorical-embedding neural net, PyTorch.

```
36 numeric ──────────────────────────────┐
q_type       → Embedding(2, 8) ─┐        │
q_blooms     → Embedding(7, 8) ─┼─ 32d ──┴─→ Linear(69→128) → ReLU → BN → Drop(0.4)
q_difficulty → Embedding(5, 8) ─┤              → Linear(128→64) → ReLU → BN → Drop(0.4)
__src        → Embedding(10,8) ─┘              → Linear(64→3)
```
`+1` = `q_loo_difficulty` (out-of-fold). Loss = class-weighted cross-entropy.

**Hyperparameters** (from exhaustive 1,728-config grid, `stage6_catemb_search.py`):
`adam · emb=8 · hidden=(128,64) · lr=0.005 · dropout=0.4 · wd=1e-4 · bs=256 · 60 epochs`

**Target — Bucketing C**: `1-2 → Low` · `3-4 → Med` · `5 → High` (9% / 41% / 50%, baseline 50.1%)

### Performance (5-fold GroupKFold by student)

| Metric | Value |
|---|---|
| Accuracy | **82.34%** (baseline 50.13%, +32.2) |
| Balanced accuracy | **0.8443** |
| Macro F1 | 0.7834 |
| Min class recall | 0.7342 |
| QWK | 0.7299 |
| AUC (OvR) | 0.9516 |

**Grid search patterns:** 19 of top 20 chose **Adam over AdamW**; 17 of 20 chose
**lr=0.005, the highest value tested** → the optimum may lie beyond the searched range.

**Finalist #2 may be the better deploy:** `adam · emb=2 · (128,64) · lr=0.005 · do=0.3 · wd=1e-3 · bs=512`
wins on 5 of 6 metrics — acc 0.8295, F1 0.7890, **min recall 0.7713**, QWK 0.7396,
AUC 0.9532 — losing only balanced acc (0.8402 vs 0.8443).

---

## 6. THE SCOPE CONDITION (most important thing in this file)

The model's top features are **leave-one-out statistics over the student's OTHER
confidence ratings in the same attempt**. Without collected ratings they do not exist.

| Ratings available | Accuracy |
|---|---|
| All (~8) | **80.3%** |
| 5 | 69.9% |
| 3 | 68.7% |
| 1 | 63.2% |
| **0 (pure cold-start)** | **47.8% — BELOW the 50.1% baseline** |

At k=0, per-quiz AUC is **0.464 / 0.493 / 0.567** — two of three are *worse than
random*. Predicting confidence from behaviour alone is **anti-predictive**, not
merely weak. This was confirmed across four architectures, six model families,
three bucketings, three engagement tiers and a 1,728-config grid.

**=> Confidence must be collected on at least some questions.**

---

## 7. Calibration-block design (the way forward)

Give students a small block of questions where confidence IS collected, then
predict the rest. Validated in `calibration_sweep.py` (randomised assignment,
8 shuffles, per-quiz, scored only on non-calibration rows).

### How many calibration questions?

| k | DBMS | DMS-2 | ELC-1 |
|---|---|---|---|
| 0 | 34.9% | 38.7% | 46.4% |
| **1** | 54.7% | 66.2% | 58.5% |
| 2 | 61.0% | 69.8% | 61.7% |
| **3** | 64.8% | **72.0%** | 63.6% |
| 5 | 66.0% | 72.5% | 64.6% |
| 7 | 67.4% | 73.0% | 65.0% |

**k=3 is the knee.** 0→3 buys +30/+33/+17 points; 3→7 buys only +2.6/+1.1/+1.4.
An earlier k=5 recommendation was **wrong** — under randomised assignment k=5 is
no better than k=3. `acc_sd` across shuffles is 0.004–0.037, so results do not
depend on which questions land in calibration.

### Which 3 questions? (`calib_difficulty_test.py`)

| Strategy | DMS-2 acc / AUC | ELC-1 acc / AUC |
|---|---|---|
| **3 medium** | **0.7118** / 0.8239 | 0.6458 / **0.7733** |
| easy/med/hard | 0.7111 / **0.8315** | 0.6407 / 0.7574 |
| 3 hardest | 0.7064 / 0.8189 | **0.6535** / 0.7642 |
| random 3 | 0.7051 / 0.8219 | 0.6332 / 0.7535 |
| **3 easiest** | **0.6927** / 0.8157 | **0.6183** / 0.7433 |

Top three are within noise. **The one robust finding: never use easy questions**
(worst on both quizzes by 1.9–2.7 pts). Recommend **3 medium-to-hard**
(~35–75% historical accuracy). Use the instructor `difficulty` label for a brand-new
quiz, empirical `percentage_correct` thereafter.

### Must the calibration items be same-subject? YES (`cross_domain_test.py`)

Using the 155 students who took 2+ quizzes:

| Measure | r |
|---|---|
| Within-subject (3 items vs rest) | **0.794** |
| Cross-subject (mean conf A vs B) | 0.557 |

Actual prediction of subject-B confidence:

| Calibration source | Acc | Baseline | AUC |
|---|---|---|---|
| Behaviour only | 52.7% | 56.0% | 0.605 |
| + **entire other subject** (~9 items) | 55.4% | 56.0% | 0.680 |
| **3 items, same subject** | **72.0%** | 48.5% | **0.830** |

**Nine questions from another subject are worth less than three from the same
subject.** A generic aptitude block (CRT-2 etc.) would transfer *worse* than
DBMS→OOPS, so expect ~55–62%, not ~72%. Not recommended.

---

## 8. What was tried and rejected

| Approach | Result |
|---|---|
| Hierarchical Bayesian ordinal + careless mixture | 45% (5-class) — target too noisy |
| Two-tower net (SBERT + option attention) | 40% — lost to hierarchical head-to-head |
| Set-transformer over attempt | dropped — order fixed, fatigue unidentifiable |
| 3-class collapsed post-hoc from 5-rung model | 45% — loses information |
| Bucketing A (1-2/3/4-5) | 70.6% acc but **71.2% baseline** — zero skill |
| **768-d SBERT embeddings** | **−0.16 acc.** ρ 0.225 random → 0.049 grouped = 78% memorisation |
| 131 unused `syn_*`/`feat_*` question columns | +0.21 — noise |
| One-hot categoricals into LightGBM | −0.23 (but **+0.45 as learned embeddings**) |
| 7 engineered interaction features | −0.13 |
| **Global PCA over design matrix** | **−10.3** — destroys the LOO signal |
| Ordinal decomposition (2 binary + monotone) | −0.9 |
| CatBoost | −1.5 (but best min-recall 0.766) |
| Soft-voting ensemble | +0.3 |
| Plain MLP (no cat embeddings) | 77.7% vs 82.2% — **embeddings are the whole gap** |

**Dead features dropped:** `q_feat_parse_tree_depth` (ρ=−0.007), `att_fast_frac` (ρ=−0.006).

---

## 9. Bugs found and fixed (do not reintroduce)

1. **Within-attempt leakage.** Under question-grouped CV, **100% of test attempts
   also appeared in training**, letting the student effect memorise that sitting's
   straightlined value. Fix: attempt-grouped split added.
2. **LOO leakage.** `conf3_pushed.py::add_loo_features` used `g.transform("std"/"min"/"max"/"nunique")`
   which **includes the row itself**. Only `mean` was true LOO. Differed from true
   LOO on 3.8% / 1.3% / 6.9% of rows → inflated the reported figure by ~11 points.
   `push_to_90.py::rich_loo` is the correct implementation. **Use that one.**
3. **`opt_sim_chosen_correct` is the label.** cos(chosen, correct) = 1.0 for 97% of
   correct answers. Gave a fake 98.8% on `is_correct`. Never use it to predict correctness.
4. **"hesitated" target is a tautology** — defined from `option_changes`/`marked_for_review`/
   `review_click_count`, all of which were in the feature set. Gave a fake 100.0%.

---

## 10. Validation protocol (non-negotiable)

- **GroupKFold by `student_uid`.** Each student has 9.4 rows on average (max 33),
  and their own mean predicts their ratings at ρ=0.719. Random K-fold reported
  ρ=0.533 where student-grouped reported ρ=0.252 — **a 2× overstatement**.
- All fold-dependent transforms (`q_loo_difficulty`, PCA, scalers) fit on the
  **training fold only**.
- Always report accuracy **next to its majority-class baseline** and
  **min-class recall** — the recurring failure mode is a class collapsing to
  ~0 recall while overall accuracy looks fine.
- Report on **clean subsets**; careless rows inflate everything.

---

## 11. Other measured results (different targets)

| Target | Accuracy | Baseline | Verdict |
|---|---|---|---|
| **Attempt-level `passed ≥40%`** | **91.5%** | 82.1% | **only genuine >90%**, RandomForest, leak-controlled |
| `is_correct` (leak-free) | 82.9% | 77.7% | real skill, AUC 0.860 |
| Answered vs unanswered (hurdle) | 91.1% | 84.9% | real, AUC 0.959 |
| Careless-attempt detection | 74.6% | 65.0% | real skill |

**On the 90% request:** confidence cannot reach it — the ceiling is ~82% and the
residual error is the Med↔High (4-vs-5) boundary that students don't apply
consistently. Only attempt-level pass/fail and engagement detection clear 90%.

---

## 12. Files

| File | Purpose |
|---|---|
| `clean_and_eda.py` | Stage 1 — cleaning + EDA → `cleaned.pkl`, `eda_report.txt` |
| `idea1_hierarchical.py` | Hierarchical Bayesian ordinal model (5-class) |
| `idea1_3class.py` | Direct 3-class hierarchical + hybrid |
| `push_to_90.py` | **`rich_loo()` lives here — the correct LOO implementation** |
| `stage5_allfeatures.py` | Unused-column / PCA / deep-learning ablation |
| `stage6_catemb_search.py` | 1,728-config CatEmbNet grid search |
| `calibration_block.py` | First calibration test (pooled — has the confounds) |
| `calibration_sweep.py` | **Corrected** per-quiz randomised k-sweep |
| `calib_difficulty_test.py` | easy/med/hard vs random vs medium-only |
| `cross_domain_test.py` | Cross-subject transfer test |
| `model_bakeoff.py` | Target × model grid (found the 91.5% pass/fail result) |
| `make_report.js` | Generates `Confidence_Prediction_Report.docx` |

Results CSVs: `stage6_final5.csv`, `push_to_90_results.csv`, `stage5_results.csv`,
`calibration_sweep.csv`, `bakeoff_results.csv`.

---

## 13. Deployment plan

**Stack:** React quiz (keep Firebase Hosting — it already works) + **DynamoDB** for
attempts (S3 is object storage, *not* a database — use it for the model file and
archival exports) + one Lambda for inference (~50ms, model ~200KB).

**Quiz flow:**
```
Q1–Q3   3 medium-to-hard questions, confidence collected (3-button widget)
Q4–Q13  no slider, behavioural telemetry only
submit  → Lambda → per-question predicted confidence + behavioural state
```

**Expected: ~65–72%** depending on quiz (not 82% — that needs full collection).
Report probabilities, not hard labels; at AUC ~0.83 the ranking is more
trustworthy than the argmax.

**Also show the behavioural state taxonomy** — a deterministic rule over data you
already log, 100% reliable, no model caveat:

| State | Rule | Share |
|---|---|---|
| Mastery | correct + no hesitation | 56.9% |
| Shaky | correct + hesitated | 20.8% |
| Misconception | wrong + no hesitation | 13.4% |
| Confusion | wrong + hesitated | 8.9% |

*hesitated = option_changes>0 OR marked_for_review OR review_click_count>0 OR above-average time*

`Bis-quiz` already logs every field needed (`timeSpent`, `optionChanges`,
`reviewClickCount`, `markedForReview`, `isCorrect`).

---

## 14. Open items

- Extend the LR sweep **above 0.005** — the grid saturated at its ceiling.
- Consider deploying **finalist #2** (better on 5 of 6 metrics).
- Export CatEmbNet to `.pt` + save `StandardScaler` params for the Lambda.
- Sensitivity-check the T2 filter thresholds (≥3 distinct values, <20% blank) —
  those are judgment calls, not derived.
- `Bis-quiz` toggles (time restriction, tab switching) verified correct; the
  time restriction only gates *starting*, not mid-exam cutoff — confirm intent.
