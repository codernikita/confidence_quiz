# Methodology — Confidence Prediction from a Calibration Block

Complete specification of the deployed model: data provenance, target
definition, every feature and how it is derived, the validation protocol, and
the known limitations. Written so that a reader can reproduce the result or
audit it without reading the source.

Reference implementations: [`scripts/fit_web_model.py`](scripts/fit_web_model.py)
(fitting) and [`src/model/predict.js`](src/model/predict.js) (inference). The two
are held in agreement by an automated parity harness (§8).

---

## 1. Problem statement

Predict a student's self-reported confidence on each quiz question, on a 1–5
scale collapsed to three ordinal classes, using behavioural telemetry plus a
small block of questions where confidence *is* collected.

**The task is not solvable without collected ratings.** Prior work on this
dataset established that from behaviour alone the model reaches 47.8% accuracy
against a 50.1% majority-class baseline — below chance-equivalent — with
per-quiz AUC of 0.464 / 0.493 / 0.567, two of three worse than random. This was
confirmed across four architectures, six model families, three target bucketings,
three engagement tiers and a 1,728-configuration hyperparameter grid.

This finding is the design constraint, not a footnote: it is why a calibration
block exists at all.

---

## 2. Data

### 2.1 Provenance

Ten Excel workbooks exported from a deployed quiz platform, each containing a
`ML_Behavioral_Data` sheet (one row per student-question), a `Question_Bank`
sheet, and a `Student_Profiles` sheet. Six distinct quizzes across three
subjects (DBMS, Discrete Mathematics, ELC).

### 2.2 Cleaning

Raw 22,975 rows → 16,990 analysable rows (1,962 attempts, 1,803 students,
107 questions). Applied in order:

| Step | Rationale | Removed |
|---|---|---|
| Drop questions absent from `Question_Bank` | No metadata available; includes an orphaned second form in `dms_quiz_one` | 534 rows / 39 attempts |
| Keep only each student's **first** attempt per subject | Repeat sittings are not independent observations | 453 attempts / 3,461 rows |
| Split `confidence == 0` into a missingness flag | The value conflated "unsure" with "did not answer" — 350 of 423 zeros in one quiz had `time_spent = 0` | 1,990 rows marked unanswered |
| Clip answered-but-rated-0 to 1 | Answered items must carry a valid rating | 297 rows adjusted |
| Deduplicate bank columns 155–165; normalise taxonomy casing (Theory/theoretical, Apply/Applying) | Schema hygiene | 11 columns |

The cleaned rating column (`conf_valid`) is NaN for unanswered items and
1–5 otherwise. The modelling frame is the 16,990 rows where it is non-null.

### 2.3 Variance decomposition

The single most important structural fact about this data:

- **53%** of confidence variance is *between students*
- **7%** is *between questions*

A student's own mean rating predicts their individual ratings at Spearman
ρ = 0.719. Who is rating matters far more than what is being rated. Every design
decision below follows from this.

---

## 3. Target

**Bucketing C**, three ordinal classes:

| Raw rating | Class | Share |
|---|---|---|
| 1–2 | Low | 9% |
| 3–4 | Medium | 41% |
| 5 | High | 50% |

Majority-class baseline: **48.73%** on the evaluated subset.

Two rejected alternatives, for the record. Bucketing A (1-2 / 3 / 4-5) reached
70.6% accuracy against a **71.2%** baseline — zero skill, an artefact of class
imbalance. Predicting all five rungs directly reached 45%; the 4-vs-5 boundary
is not applied consistently by students and is largely irreducible noise.

---

## 4. The calibration block

### 4.1 Why three items

Measured directly by randomised assignment (`calibration_sweep.py`), 8 shuffles,
scored only on non-calibration rows:

| k | DBMS | DMS-2 | ELC-1 |
|---|---|---|---|
| 0 | 34.9% | 38.7% | 46.4% |
| 1 | 54.7% | 66.2% | 58.5% |
| 2 | 61.0% | 69.8% | 61.7% |
| **3** | **64.8%** | **72.0%** | **63.6%** |
| 5 | 66.0% | 72.5% | 64.6% |
| 7 | 67.4% | 73.0% | 65.0% |

Going 0→3 buys +30 / +33 / +17 points. Going 3→7 buys +2.6 / +1.1 / +1.4.
**k = 3 is the knee.** Standard deviation across shuffles is 0.004–0.037, so the
result does not depend on which particular questions land in the block.

An earlier k=5 recommendation was withdrawn: under randomised assignment k=5 is
not distinguishable from k=3.

### 4.2 Which three items

| Strategy | DMS-2 acc / AUC | ELC-1 acc / AUC |
|---|---|---|
| 3 medium | **0.7118** / 0.8239 | 0.6458 / **0.7733** |
| easy / medium / hard | 0.7111 / **0.8315** | 0.6407 / 0.7574 |
| 3 hardest | 0.7064 / 0.8189 | **0.6535** / 0.7642 |
| random 3 | 0.7051 / 0.8219 | 0.6332 / 0.7535 |
| **3 easiest** | **0.6927** / 0.8157 | **0.6183** / 0.7433 |

The top three strategies are within noise of each other. The one robust finding
is negative: **never use easy items** — worst on both quizzes by 1.9–2.7 points.
Easy items compress ratings toward 5 and yield no within-student variance to
learn from.

Deployed choice: three medium-to-hard items.

### 4.3 Domain transfer — the principal limitation

Using the 155 students who took two or more quizzes:

| Measure | r |
|---|---|
| Within-subject (3 items vs the rest) | **0.794** |
| Cross-subject (mean confidence A vs B) | 0.557 |

Predicting subject-B confidence:

| Calibration source | Accuracy | Baseline | AUC |
|---|---|---|---|
| Behaviour only | 52.7% | 56.0% | 0.605 |
| + entire other subject (~9 items) | 55.4% | 56.0% | 0.680 |
| **3 items, same subject** | **72.0%** | 48.5% | **0.830** |

Nine questions from another subject are worth less than three from the same
subject.

**The deployed application calibrates on general reasoning items and predicts
technical items.** This is a cross-domain configuration. The measured figures in
§7 come from within-subject calibration and should be read as an upper bound;
expect several points below them in deployment. This is disclosed to the student
on the results page rather than suppressed.

---

## 5. Features

23 features, all computable in a browser from telemetry the quiz already logs.
Names, order, clipping and standardisation are identical in
`fit_web_model.py::engineer` and `predict.js::buildFeatures`.

### 5.1 Calibration block — 7 features

Leave-one-out statistics over the collected ratings. Let `R` be the multiset of
calibration ratings, and for a calibration question *i*, `R₋ᵢ` be `R` with that
question's own rating removed.

| Feature | Definition | Applied to |
|---|---|---|
| `calib_mean` | mean(R₋ᵢ) | calibration rows |
| | mean(R) | technical rows |
| `calib_std` | population SD (ddof = 0) | as above |
| `calib_min` | min | as above |
| `calib_max` | max | as above |
| `calib_range` | max − min | as above |
| `calib_high_frac` | fraction equal to 5 | as above |
| `calib_low_frac` | fraction ≤ 2 | as above |

The leave-one-out construction is what makes the calibration rows honest: a
calibration question is scored without access to the answer it is being scored
against. Technical questions contributed no rating, so they see all three.

> **A prior bug, documented so it is not reintroduced.** An earlier
> implementation built these with pandas `groupby.transform("std"/"min"/"max")`,
> which **includes the row itself**. Only `transform("mean")` happens to be true
> LOO. The leak differed from correct LOO on 1.3–6.9% of rows and inflated the
> reported figure by roughly 11 points.

### 5.2 Per-question behaviour — 9 features

| Feature | Definition |
|---|---|
| `is_correct` | 1 if the selected option equals the correct answer |
| `log_time` | `log(1 + clip(time_spent, 0, 900))`, seconds |
| `time_rel_attempt` | `log_time − mean(log_time)` over that attempt |
| `option_changes` | `clip(count, 0, 10)`; first selection is not a change |
| `any_option_change` | 1 if `option_changes > 0` |
| `marked_for_review` | 1 if flagged |
| `review_click_count` | `clip(count, 0, 10)` |
| `hesitated` | 1 if `option_changes > 0` **or** `marked_for_review` **or** `review_click_count > 0` **or** `time_rel_attempt > 0` |
| `q_position_frac` | position index ÷ (items − 1), in [0, 1] |

Clipping bounds were chosen to bound outliers, not tuned: 900 s is 15 minutes on
one question, and 10 is far into the tail of both counters.

`time_rel_attempt` is centred *within the attempt*, so it measures deliberation
relative to that student's own pace rather than an absolute threshold. This
matters because engaged and disengaged students at the same wall-clock speed
behave oppositely: at 3–5 seconds, engaged students are 86.7% correct (mastery)
and disengaged students 15.4% (abandonment). Absolute time cannot separate these;
attempt-relative time can.

**Telemetry fidelity.** `time_spent` accrues only while the browser tab is
visible and focused. Without this a student who switches away registers as deep
deliberation, and time is a model input.

### 5.3 Question metadata — 2 features

| Feature | Definition |
|---|---|
| `q_difficulty_num` | instructor label mapped `{very easy:1, easy:2, medium/moderate:3, hard/difficult:4, very hard:5}`, default 3 |
| `q_is_theory` | 1 if the type string contains "theor" |

Both carry almost no weight (§7.2), consistent with only 7% of variance being
between questions. They are retained because they cost nothing and because their
near-zero coefficients are themselves a reportable result.

### 5.4 Attempt and student context — 5 features

| Feature | Definition |
|---|---|
| `att_log_total_time` | `log(1 + Σ time_spent)` over the attempt |
| `att_unans_frac` | fraction of items with no option selected |
| `att_mean_changes` | mean `option_changes` over the attempt |
| `att_n_items` | number of items in the attempt |
| `student_cgpa` | 0–10, median-imputed when missing |

CGPA is collected as optional. A missing value is standardised to 0, i.e. the
training mean, which is the neutral choice — and given its weight (§7.2) the cost
of omitting it is negligible.

### 5.5 Standardisation

All features are z-scored with a `StandardScaler` fitted on the training fold
only. The exported `scaler.mean` and `scaler.scale` vectors are the average
across the five folds. `predict.js` applies the identical transform, and
substitutes 0 (the training mean) for any absent value.

---

## 6. Model

Two heads over the same standardised feature vector.

**Multinomial logistic regression** → P(Low), P(Medium), P(High).
`solver=lbfgs, C=1.0, class_weight="balanced", max_iter=2000`.
`class_weight="balanced"` is load-bearing: the Low class is 9% of the data and
collapses to near-zero recall without it.

**Ridge regression** → the continuous 1–5 rating. `alpha=5.0`.

The value plotted for the student is the mean of the two heads:

```
confidence = 0.5 · ridge + 0.5 · Σₖ P(classₖ) · centreₖ      centres = [1.5, 3.5, 5.0]
```

The uncertainty band is the SD of that class distribution:

```
sd = √( Σₖ P(classₖ) · (centreₖ − expected)² )
```

so the band widens exactly where the class probabilities are most spread —
it is the model's own uncertainty, not a fixed interval.

**Why a linear model.** The best model found on this dataset is CatEmbNet, a
categorical-embedding neural network reaching 82.34% accuracy under full
rating collection. It is not deployed here. The application has no inference
server — the backend is Firestore only — so the model must evaluate in
JavaScript. Logistic regression exports as a 3 × 23 weight matrix (5,960 bytes
of JSON) and scores a 15-question attempt in **83 microseconds**, ~1,380
floating-point operations. The accuracy cost of that constraint is real and is
reported rather than hidden.

---

## 7. Validation

### 7.1 Protocol

- **5-fold `GroupKFold` grouped by `student_uid`.** Non-negotiable: students
  average 9.4 rows each and their own mean predicts their ratings at ρ = 0.719,
  so any split that puts one student on both sides leaks. Measured directly —
  random K-fold reported ρ = 0.533 where student-grouped reported ρ = 0.252, a
  **two-fold overstatement**.
- **8 independent random calibration draws** (seeds 17–24), 3 items per attempt.
- **Scored only on non-calibration rows.** Calibration rows are inputs, not test
  cases; including them would score the model on questions whose answer it was
  given.
- All fold-dependent transforms (scaler, and the calibration statistics) are
  fitted on the training fold only.
- Accuracy is always reported beside its majority-class baseline and beside
  min-class recall. The recurring failure mode on this dataset is a class
  collapsing to ~0 recall while overall accuracy looks acceptable.

### 7.2 Results

Mean over 8 shuffles; standard deviations in parentheses.

| Metric | Value |
|---|---|
| Accuracy | **0.6374** (0.0037) |
| Majority-class baseline | 0.4873 (0.0010) |
| **Lift** | **+15.01 points** |
| Balanced accuracy | 0.6340 (0.0028) |
| Macro F1 | 0.5841 (0.0031) |
| Min-class recall | 0.5473 (0.0080) |
| Quadratic weighted κ | 0.5127 (0.0054) |
| AUC (OvR, macro) | 0.8143 (0.0019) |

Standardised coefficients, ranked by largest absolute weight across classes:

| # | Feature | \|w\|max | | # | Feature | \|w\|max |
|---|---|---|---|---|---|---|
| 1 | `calib_high_frac` | 0.599 | | 13 | `marked_for_review` | 0.106 |
| 2 | `calib_mean` | 0.479 | | 14 | `q_position_frac` | 0.101 |
| 3 | `calib_std` | 0.384 | | 15 | `att_n_items` | 0.072 |
| 4 | `is_correct` | 0.260 | | 16 | `review_click_count` | 0.068 |
| 5 | `calib_max` | 0.259 | | 17 | `att_unans_frac` | 0.048 |
| 6 | `calib_range` | 0.225 | | 18 | `q_is_theory` | 0.047 |
| 7 | `calib_min` | 0.216 | | 19 | `hesitated` | 0.046 |
| 8 | `calib_low_frac` | 0.159 | | 20 | `student_cgpa` | 0.044 |
| 9 | `log_time` | 0.159 | | 21 | `att_mean_changes` | 0.021 |
| 10 | `att_log_total_time` | 0.136 | | 22 | `q_difficulty_num` | 0.018 |
| 11 | `any_option_change` | 0.124 | | 23 | `option_changes` | 0.015 |
| 12 | `time_rel_attempt` | 0.109 | | | | |

The top eight are seven calibration features plus `is_correct`. This is the
scope condition (§1) appearing in the weights: the collected ratings do the work
and behaviour adjusts around them.

Two secondary observations. The binary `any_option_change` (0.124) carries eight
times the weight of the raw count `option_changes` (0.015) — *whether* a student
revised matters, *how many times* does not. And `calib_std` pushes toward
**Medium**: a student who uses the full scale is one the model declines to place
at an extreme.

### 7.3 Measured sensitivity

Effect on predicted confidence of varying one input, all else held fixed:

| Input varied | Range | Δ confidence |
|---|---|---|
| Calibration ratings [1,1,2] → [5,5,5] | full | **2.62** |
| Correct vs wrong | binary | 0.43 |
| Time on question, 5 s → 400 s | full | 0.20 |
| Difficulty label, very easy → very hard | full | 0.16 |
| Difficulty label, within the deployed bank | medium → hard | **0.04** |

---

## 8. Implementation fidelity

The fitting code (Python/scikit-learn) and the inference code (JavaScript) are
separate implementations of one model and will drift silently unless checked.
`scripts/parity_test.mjs` runs a fixed synthetic attempt through the browser
code; `scripts/parity_check.py` recomputes it in NumPy and diffs every
intermediate — attempt aggregates, leave-one-out statistics, all 23 standardised
features, class probabilities, both heads.

Current status: agreement to **1e-9** on every value. The check is re-run after
any change to either side.

---

## 9. Limitations

1. **Cross-domain calibration (§4.3).** The largest one. Reasoning items
   calibrating technical predictions transfer at r = 0.557 rather than 0.794.
   Fixed by swapping the calibration block to same-subject items; no code change
   required.

2. **Ceiling.** Confidence cannot reach 90% on this data. The ceiling is ~82%
   with *full* rating collection, and the residual error is concentrated at the
   Medium↔High (4-vs-5) boundary, which students do not apply consistently. Only
   attempt-level pass/fail (91.5%) and engagement detection clear 90%.

3. **`att_n_items` distribution shift.** Training attempts averaged 9.89 rated
   items (SD 2.84); the deployed quiz always has 15, a z-score of +1.80. This
   applies a **constant** offset to every prediction (−0.032 confidence points on
   the ridge head, +0.11/−0.13 logit on Low/High). Because it is identical for
   every question in an attempt, it shifts levels but not the ordering across
   questions — which is the reading the interface directs users toward. Refitting
   on 15-item attempts would remove it.

4. **`att_unans_frac` and `att_n_items` are computed differently in the two
   implementations.** In training they come from the source attempt records; in
   the browser they are computed over presented items. Both measure the same
   quantity but not identically. Combined weight is 0.12, so the effect is small,
   but it is a genuine fidelity gap rather than a rounding difference.

5. **Straightlining inflates pooled metrics.** 38.5% of attempts in the source
   data were straightlined. Constant-target rows are trivially predictable: the
   same model scored ρ = 0.44 on straightlined rows against ρ = 0.29 on clean
   rows. Reported figures are not stratified by engagement tier; a per-tier
   breakdown would show lower numbers on the engaged subset.

6. **Self-reported confidence is the ground truth.** The target is what students
   *say*, not a calibrated measure of what they know. `is_correct` is a feature,
   not the label.

7. **Single institution, six quizzes, three subjects.** External validity beyond
   this population is untested.

8. **Small-sample per-student LOO.** The mean absolute error over three held-out
   calibration ratings, shown to each student, is computed on n = 3. It is a
   reliability hint, not a statistic, and the interface describes it as such.

---

## 10. Reproduction

```bash
conda activate badminton
python scripts/fit_web_model.py                 # → src/model/coefficients.json
node scripts/parity_test.mjs > /tmp/js.json
python scripts/parity_check.py /tmp/js.json     # must report PARITY OK
```

Deterministic given `cleaned.pkl`: seeds are fixed (17 + shuffle index), and
`GroupKFold` is not randomised. `coefficients.json` embeds the metrics of the run
that produced it, so any deployed model can be traced back to its own validation
numbers.
