# METHODOLOGY

How the confidence model was built, why leave-one-out is the centre of it, and
what the evidence for each decision was.

Every number here was measured on this dataset. Nothing is estimated or quoted
from literature unless explicitly marked as such.

---

## Contents

1. [The problem](#1-the-problem)
2. [Data and cleaning](#2-data-and-cleaning)
3. [The finding that determined everything](#3-the-finding-that-determined-everything)
4. [Leave-one-out: the core method](#4-leave-one-out-the-core-method)
5. [Why GroupKFold](#5-why-groupkfold)
6. [Target definition](#6-target-definition)
7. [Model selection](#7-model-selection)
8. [What failed](#8-what-failed)
9. [The cold-start wall](#9-the-cold-start-wall)
10. [Calibration-block design](#10-calibration-block-design)
11. [Leakage bugs found](#11-leakage-bugs-found)
12. [Results summary](#12-results-summary)

---

## 1. The problem

Predict a student's **confidence** on each quiz question — bucketed Low /
Medium / High — from behavioural telemetry (time spent, option changes, review
clicks, correctness) plus question metadata.

The application is a results page that shows, per question, both whether the
student was right and how confident they were, so a teacher can distinguish
*"got it wrong and knew they were guessing"* from *"got it wrong while certain"*.

---

## 2. Data and cleaning

Ten Excel workbooks, one per quiz. Each has `ML_Behavioral_Data` (one row per
student-question response), `Question_Bank`, and `Student_Profiles`.

`clean_and_eda.py` reduced **22,975 raw rows → 16,990** usable rows
(1,962 attempts, 1,803 students, 107 questions):

| Fix | Rationale | Removed |
|---|---|---|
| Drop questions absent from `Question_Bank` | Includes an orphan second quiz form embedded in `dms_quiz_one` — nothing is known about those items | 534 rows / 39 attempts |
| **Keep only each student's FIRST attempt per subject** | Some students re-sat the same quiz up to 11 times. By attempt 8 their confidence reflects *memory*, not first-encounter understanding | **453 attempts / 3,461 rows** |
| Split `confidence == 0` | It conflated "totally unsure" with "never answered". 350 of 423 zeros in one file were unanswered items with `time_spent = 0` | 1,990 rows reflagged |
| Clip answered-but-rated-0 → 1 | Genuine rating, invalid value | 297 rows |
| De-duplicate bank columns 155–165; normalise taxonomy casing | `Theory`/`theoretical` and `Apply`/`Applying` were being treated as distinct categories | 11 columns |

The repeat-sitting fix matters more than its row count suggests: it removes the
subset of the data where the target means something different.

---

## 3. The finding that determined everything

Before modelling, we decomposed the variance in `confidence_rating` by grouping
level:

| Grouping | Groups | Between-group variance share |
|---|---|---|
| **`student_uid`** | 1,873 | **53.2%** |
| `attempt_id` | 2,315 | 57.4% |
| `question_id` | 125 | **7.0%** |
| Source quiz | 10 | 4.2% |

**More than half the variation in confidence ratings is between students, and
only 7% is between questions.**

Some students click 5 on everything; others are habitually cautious. That
personal *response style* dominates any property of the question being asked.
Confirming it directly: a student's own mean rating predicts any individual
rating of theirs at **Spearman ρ = 0.719** — far stronger than any behavioural
feature we measured (best was `is_correct` at ρ = 0.195).

This reframed the task. The question stopped being *"what about this question
makes a student confident?"* and became *"what kind of rater is this person, and
where does this question sit relative to their baseline?"*

Everything downstream follows from that.

---

## 4. Leave-one-out: the core method

### 4.1 The obvious idea, and why it fails

To exploit the 53%, the model needs a feature describing the student's rating
style. The naive version:

> For each row, `student_mean` = the average of **all** that student's ratings.

This is **target leakage**. Row 7's own rating is inside its own feature. A
student who rated `[5,5,5,5,5,5,5]` gets `student_mean = 5`, and the model
learns "predict the mean" — scoring beautifully in validation and collapsing in
production, where the answer isn't available.

### 4.2 The fix

Compute every statistic from the student's **other** ratings, excluding the
current row:

```
Student rated:   Q1=5   Q2=4   Q3=5   Q4=3   Q5=5

row Q1 → loo_conf_mean = (4+5+3+5)/4 = 4.25    ← Q1's own 5 excluded
row Q2 → loo_conf_mean = (5+5+3+5)/4 = 4.50    ← Q2's own 4 excluded
row Q3 → loo_conf_mean = (5+4+3+5)/4 = 4.25
```

Every row gets a different value, and no row's label ever contributes to its own
features.

### 4.3 Computing it efficiently

Recomputing a mean per row is O(n²). Instead, subtract your own contribution
from the group total:

```python
s = group.transform("sum")            # total for the attempt
n = group.transform("count")
loo_mean = (s - own_rating) / (n - 1)  # exact, single pass
```

**This shortcut only works for sums.** `min`, `max`, `std` and `nunique` cannot
be "un-added" — they must be recomputed over the actual complement. Missing
that is exactly the bug described in §11.1.

### 4.4 The feature block

First implementation used 6 statistics. Expanding to **19** — capturing the
*shape* of the rating distribution rather than just its centre — was the single
largest gain in the project (**+12 to +14 points**).

| Feature | Definition | Spearman vs target |
|---|---|---|
| `loo_conf_mean` | mean of their other ratings | **+0.636** |
| `loo_frac_5` | **fraction of their others that were 5** | **+0.633** |
| `loo_stu_mean` | mean across their entire record | +0.625 |
| `loo_q25` | lower quartile | +0.592 |
| `loo_median` | median | +0.589 |
| `loo_q75` | upper quartile | +0.551 |
| `loo_mode` | most common other rating | +0.541 |
| `loo_conf_min` | lowest other rating | +0.529 |
| `loo_frac_3` | fraction that were 3 | −0.430 |
| `loo_conf_max` | highest other rating | +0.400 |
| `loo_conf_nuniq` | count of distinct values used | −0.388 |
| `loo_mode_share` | share of ratings at the mode | +0.379 |
| `loo_stu_sd` | spread, whole record | −0.376 |
| `loo_conf_sd` | spread, this attempt | −0.357 |
| `loo_frac_2` | fraction that were 2 | −0.355 |
| `loo_iqr` | interquartile range | −0.308 |
| `loo_frac_4` | fraction that were 4 | −0.307 |
| `loo_frac_1` | fraction that were 1 | −0.283 |
| `loo_n` | how many other ratings exist | −0.170 |

Mean |ρ| for this block is **0.445**, versus **0.075** for all 17
behaviour/question features combined — a **6× gap**. `loo_conf_mean` alone beats
the best non-LOO feature by more than 3×.

`loo_frac_5` at +0.633 justifies the expansion on its own: *"what share of their
ratings are 5s"* is nearly as predictive as the mean and expresses something the
mean cannot.

### 4.5 Reading the negative correlations

The negatives all say one thing: **students who use more of the scale rate
lower**. A student with `nuniq = 1` clicked one value throughout (almost always
5); a student with `nuniq = 5` genuinely discriminates, and their average is
lower. `loo_frac_4` being negative (−0.307) is the same effect — heavy raters
park on 5, not 4, so being a frequent-4-giver signals a more critical rater.

`loo_conf_sd = 0` is also the straightliner fingerprint, which is how the model
learns to distinguish a genuinely confident student from an unengaged one.

### 4.6 The same trick, applied to questions

`q_loo_difficulty` = the fraction of **other students** who answered this
question correctly, recomputed **inside each training fold**. Same principle:
the row's own outcome never contributes to its own difficulty estimate, and
test-fold outcomes never leak into the encoding.

---

## 5. Why GroupKFold

Ordinary K-fold is invalid here, and we measured how badly.

Each student contributes **9.4 rows on average, up to 33**. Random splitting
scatters those rows across folds:

```
ORDINARY K-FOLD — student S has 10 rows
  TRAIN:  S-Q1 S-Q2 S-Q3 S-Q4 S-Q5 S-Q6 S-Q7 S-Q8
  TEST:                                     S-Q9 S-Q10
                                              ↑
  The model already saw 8 of S's ratings. It memorises "S is a 5-rater"
  and predicts 5. Scores well. Learned nothing transferable.
```

Since a student's own mean predicts their ratings at ρ = 0.719, this is a large
and easy shortcut. **GroupKFold by `student_uid`** puts all of a student's rows
on the same side of every split, so the model always faces people it has never
seen — which is the real deployment condition.

Measured on identical data and model:

| Split | Spearman |
|---|---|
| Random K-fold (leaky) | 0.533 |
| GroupKFold by `attempt_id` | 0.289 |
| **GroupKFold by `student_uid`** | **0.252** |

**Random splitting overstates performance by roughly 2×.**

Grouping by attempt is insufficient: a student with two sittings could still
have one in train and one in test, leaking their style. A third variant,
GroupKFold by `question_id`, answers a *different* question — "can we predict a
brand-new question?" — and is what exposed the embedding failure in §8.

**All fold-dependent transforms** (`q_loo_difficulty`, PCA, scalers) are fit on
the training fold only.

---

## 6. Target definition

The raw scale is 1–5. Three bucketings were tested:

| Bucketing | Mapping | Balance | Baseline | Verdict |
|---|---|---|---|---|
| A | 1-2 / 3 / 4-5 | 9 / 15 / 76% | 75.7% | **Rejected** — 70.6% accuracy against a 71.2% baseline, i.e. *worse than always guessing "High"*. Min-class recall 0.004 |
| B | 1-3 / 4 / 5 | 24 / 26 / 50% | 50.1% | Viable, slightly weaker |
| **C** | **1-2 / 3-4 / 5** | 9 / 41 / 50% | **50.1%** | **Selected** — best accuracy with all three classes actually predicted |

Bucketing A is the cautionary case: it produces the highest raw accuracy of the
three while having *no skill at all*. Every accuracy in this project is therefore
reported next to its majority-class baseline **and** its minimum per-class
recall.

Two further decisions:

- **Train directly on 3 classes.** Fitting a 5-rung model and collapsing its
  probabilities afterward lost information: QWK 0.20 → 0.26 and macro-F1
  0.40 → 0.43 when trained directly.
- **Class-weighted loss.** Without it, the middle class repeatedly collapsed to
  ~0 recall while overall accuracy still looked acceptable.

---

## 7. Model selection

The research model is **CatEmbNet** — a small PyTorch network with learned
embeddings for the categorical fields.

```
36 numeric ─────────────────────────────────┐
q_type       → Embedding(2,  8) ─┐          │
q_blooms     → Embedding(7,  8) ─┼─ 32d ────┴─→ Linear(69→128) → ReLU → BN → Drop(0.4)
q_difficulty → Embedding(5,  8) ─┤               → Linear(128→64) → ReLU → BN → Drop(0.4)
__src        → Embedding(10, 8) ─┘               → Linear(64→3)
```

Hyperparameters from an **exhaustive 1,728-configuration grid**
(`stage6_catemb_search.py`), screened at 3-fold and the top 8 re-scored at full
5-fold: `adam · emb=8 · (128,64) · lr=0.005 · dropout=0.4 · wd=1e-4 · bs=256`.

Selection was on **balanced accuracy**, not raw accuracy, to stop the search
optimising into a class collapse.

### Why a neural net won here — and it's a narrow reason

| Model | Accuracy |
|---|---|
| **CatEmbNet** | **82.34%** |
| LightGBM (tuned) | 82.21% |
| LightGBM (stock) | 81.52% |
| Ordinal-LGBM (2 binary + monotone) | 81.14% |
| CatBoost | 80.46% |
| **Plain MLP, no categorical embeddings** | **77.73%** |
| Logistic regression | 70.94% |

On raw accuracy CatEmbNet and LightGBM are tied (+0.13). CatEmbNet wins on
**balanced accuracy (0.844 vs 0.780)** and **min-class recall (0.734 vs 0.667)**.

The mechanism is visible in two rows:

- LightGBM + the categoricals **one-hot**: 81.77% — *worse* than without them
- CatEmbNet + the same categoricals as **learned embeddings**: 82.22%

Same information, opposite outcome. And the plain-MLP row is the control: same
architecture family, same data, **the entire 4.5-point gap is the embeddings**.
That is the specific, narrow case where neural nets beat trees on tabular data.

Two grid patterns worth recording: **19 of the top 20** configurations chose
Adam over AdamW, and **17 of 20** chose the highest learning rate tested
(0.005) — suggesting the optimum lies beyond the searched range.

---

## 8. What failed

Recorded because the negative results were as informative as the positive ones.

| Approach | Measured effect |
|---|---|
| **768-d SBERT question embeddings** | **−0.16 accuracy.** ρ 0.225 under random splits → **0.049** under question-grouped splits. ~78% of the apparent signal was memorised question identity: with only 92 embeddable questions against 16,990 rows, a 768-d vector functions as a lookup key |
| PCA-8 of those embeddings | Recovered most of the damage but never beat plain behaviour |
| 131 unused `syn_*` / `feat_*` question columns (POS, dependency, parse stats) | +0.21 — noise |
| One-hot categoricals into LightGBM | −0.23 |
| 7 engineered interaction terms | −0.13 |
| **Global PCA over the whole design matrix** | **−10.3** — blends the load-bearing LOO features in with 130+ near-useless columns |
| Ordinal decomposition (2 binary + monotonicity) | −0.9 |
| Soft-voting ensemble | +0.3 |
| Two-tower net (SBERT + option attention) | 40% on the 5-class target; lost head-to-head to a hierarchical model |
| Hierarchical Bayesian ordinal + careless mixture | 45% on the 5-class target |
| Set-transformer over the attempt | Not run — question order is fixed (1–4 distinct orders per workbook) and there are no per-item timestamps, so fatigue/order effects are unidentifiable |

Dropped as statistically zero: `q_feat_parse_tree_depth` (ρ = −0.007) and
`att_fast_frac` (ρ = −0.006).

**The embedding result is the headline negative.** Ten separate experiments
tried to make question text useful; none did. `q_loo_difficulty` already captures
"this item is hard" empirically, with far less overfitting risk.

---

## 9. The cold-start wall

The LOO block requires the student's other ratings. So: what happens without
them? This is the most important measurement in the project.

| Ratings available | Accuracy |
|---|---|
| All (~8) | **80.3%** |
| 5 | 69.9% |
| 3 | 68.7% |
| 1 | 63.2% |
| **0 (pure cold-start)** | **47.8%** |

The baseline is 50.1%. **At k=0 the model is below baseline.** Per-quiz AUC at
k=0 was **0.464 / 0.493 / 0.567** — two of three are *worse than random ranking*.

This is not a weak model. It is **anti-predictive**. Confidence cannot be
predicted from behaviour alone on this data, and that conclusion survived:

- 4 architectures (hierarchical Bayesian, two-tower, CatEmbNet, boosting)
- 6 model families
- 3 bucketings
- 3 engagement-filter tiers
- a 1,728-configuration grid
- and single-quiz replication on three separate quizzes

**Consequence:** confidence must be collected on *some* questions. That is not a
design preference; it is what makes the feature exist at all.

---

## 10. Calibration-block design

The deployed solution: collect confidence on a small block of questions, predict
the rest.

### 10.1 How many questions?

`calibration_sweep.py`, run **per quiz** with calibration items assigned **at
random per attempt** and averaged over **8 shuffles**, scored **only on
non-calibration rows**:

| k | DBMS | DMS-2 | ELC-1 |
|---|---|---|---|
| 0 | 34.9% | 38.7% | 46.4% |
| **1** | 54.7% | 66.2% | 58.5% |
| 2 | 61.0% | 69.8% | 61.7% |
| **3** | 64.8% | **72.0%** | 63.6% |
| 5 | 66.0% | 72.5% | 64.6% |
| 7 | 67.4% | 73.0% | 65.0% |

**k = 3 is the knee.** 0→3 buys +30 / +33 / +17 points; 3→7 buys only
+2.6 / +1.1 / +1.4. `acc_sd` across shuffles was 0.004–0.037, so the result does
not depend on *which* questions land in calibration.

An earlier recommendation of k=5 was **wrong** — it came from a pooled test
whose "first-k" selection was confounded with question identity (order is fixed
per workbook, so "first 5" was always the same 5 questions).

### 10.2 Which questions?

`calib_difficulty_test.py`:

| Strategy | DMS-2 acc / AUC | ELC-1 acc / AUC |
|---|---|---|
| 3 medium | **0.7118** / 0.8239 | 0.6458 / **0.7733** |
| easy / medium / hard | 0.7111 / **0.8315** | 0.6407 / 0.7574 |
| 3 hardest | 0.7064 / 0.8189 | **0.6535** / 0.7642 |
| random 3 | 0.7051 / 0.8219 | 0.6332 / 0.7535 |
| **3 easiest** | **0.6927** / 0.8157 | **0.6183** / 0.7433 |

The top three are within noise. **The one robust finding: never use easy
questions** — worst on both quizzes by 1.9 and 2.7 points. Easy items draw
uniformly high confidence from nearly everyone, so they don't discriminate a
genuinely confident student from a habitually confident one. This is the
*hard-easy effect* from the metacognition literature (overconfidence grows with
difficulty; underconfidence appears on easy items).

Recommendation: **3 medium-to-hard items** (~35–75% historical accuracy). Use
the instructor `difficulty` label for a brand-new quiz, empirical
`percentage_correct` thereafter.

### 10.3 Must they be from the same subject? Yes.

`cross_domain_test.py`, using the 155 students who took 2+ quizzes:

| Measure | r |
|---|---|
| Within-subject (3 items vs the rest) | **0.794** |
| Cross-subject (mean confidence A vs B) | 0.557 |

And in actual prediction of subject-B confidence:

| Calibration source | Accuracy | Baseline | AUC |
|---|---|---|---|
| Behaviour only | 52.7% | 56.0% | 0.605 |
| + **the student's entire other subject** (~9 items) | 55.4% | 56.0% | 0.680 |
| **3 items, same subject** | **72.0%** | 48.5% | **0.830** |

**Nine rated questions from another subject are worth less than three from the
subject at hand.** Confidence is domain-specific. A generic aptitude block
(CRT-2, Berlin Numeracy) is a *further* domain jump than DBMS→OOPS, so it would
transfer at or below 0.557 — expect ~55–62%, not ~72%. Not recommended.

---

## 11. Leakage bugs found

Both were caught by results that were implausibly good. Documented so they are
not reintroduced.

### 11.1 LOO statistics that included the row itself

The first implementation (`legacy/conf3_pushed_LEAKY.py.txt`):

```python
a["loo_conf_mean"]  = (s - a.conf_valid) / (n - 1)   # correct LOO
a["loo_conf_sd"]    = g.transform("std")             # ← includes the row
a["loo_conf_max"]   = g.transform("max")             # ← includes the row
a["loo_conf_min"]   = g.transform("min")             # ← includes the row
a["loo_conf_nuniq"] = g.transform("nunique")         # ← includes the row
```

Only the mean was genuinely leave-one-out. The sum-subtraction shortcut does not
generalise to order statistics (§4.3), and `transform()` operates over the whole
group.

Measured divergence from true LOO: **3.8%** of rows for `min`, **1.3%** for
`max`, **6.9%** for `nunique`. It **inflated the reported figure by ~11 points**
(80.6% leaky vs 69.8% honest on the same 6-feature set).

The correct implementation is `push_to_90.py::rich_loo()`, which recomputes each
statistic over the actual complement. The clean 19-feature version (82.0%)
subsequently *exceeded* the leaky 6-feature one.

The leaky file is retained as `.py.txt` so `import` cannot reach it.

### 11.2 A feature that was the label

While testing `is_correct` as an alternative target, accuracy came back at
**98.8% with AUC 0.999**. The cause: `opt_sim_chosen_correct` =
cosine(chosen_option, correct_answer), which equals **exactly 1.0 for 97% of
correct answers**. The feature *was* the label. Leak-free, that target scores
82.9%.

### 11.3 A tautological target

A "hesitated" target returned **exactly 100.0%**. It is *defined* from
`option_changes` / `marked_for_review` / `review_click_count` — all of which were
in the feature set. Discarded.

### 11.4 Within-attempt leakage under question-grouped CV

Checked directly: under GroupKFold by `question_id`, **100% of test attempts
also appeared in training**, letting the student effect memorise that sitting's
straightlined value. Careless-row MAE dropped to an implausible 0.16. An
attempt-grouped split was added and affected figures re-measured.

---

## 12. Results summary

### Research model (all ratings collected)

| Metric | Value |
|---|---|
| Accuracy | **82.34%** (baseline 50.13%, **+32.2**) |
| Balanced accuracy | 0.8443 |
| Macro F1 | 0.7834 |
| Min class recall | 0.7342 |
| Quadratic weighted kappa | 0.7299 |
| AUC (one-vs-rest) | 0.9516 |

### Progression

| Stage | Accuracy |
|---|---|
| 5-class model collapsed post-hoc | 45.0% |
| Direct 3-class training | 45.0% |
| **+ leave-one-out features** | **80.6%** |
| + rich 19-feature LOO block (clean) | 82.0% |
| + categorical embeddings | 82.2% |
| + exhaustive grid search | 82.3% |

**Leave-one-out accounts for essentially the entire gain.** No architecture
change, feature-engineering effort, or hyperparameter search contributed more
than 1.5 points.

### Deployed model (3-question calibration block)

**~66–72%** depending on the quiz, AUC 0.76–0.83. Multinomial logistic + ridge
over 23 browser-computable features, exported as JSON weights, because Firestore
is the only backend and there is no inference server.

### Honest ceiling

82% is close to the limit for this target on this data. The residual error is
concentrated on the **Medium↔High boundary** — a 4-versus-5 distinction that
students do not apply consistently (81.3% vs 86.6% accuracy, the smallest
adjacent gap on the scale). Reaching 90% would mean halving the error on a
boundary the instrument itself does not cleanly separate. The realistic fix is a
coarser collection scale, not a better classifier.

For reference, targets on this data that *do* clear 90%: attempt-level pass/fail
(91.5%, leak-controlled) and answered-vs-unanswered (91.1%, AUC 0.959).

### A note on what this model is

It predicts confidence **given some collected ratings** — an imputation and
anomaly-detection tool, not a cold-start forecaster. Reported that way, 82%
(research) and ~70% (deployed) are defensible. Reported as cold-start
prediction, they would be wrong by construction, since that setting measures
47.8% — below baseline.
