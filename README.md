# Confidence Calibration Quiz

A quiz that collects confidence ratings on **three** reasoning questions, then
**predicts** confidence on every technical question that follows — and shows the
student a chart of the result.

React + Vite. **Firestore is the only backend**; the model runs in the browser.
Deploys to AWS as a static site (S3 + CloudFront) — see [DEPLOY_AWS.md](DEPLOY_AWS.md).

---

## Why it is built this way

The research behind this (`../CONTEXT.md`) found one thing that dictates the
whole design:

> Confidence **cannot** be predicted from behaviour alone. With no collected
> ratings the model scores **47.8%** against a **50.1%** majority-class
> baseline — worse than guessing the most common class. Per-quiz AUC at k=0 is
> 0.464 / 0.493 / 0.567; two of three are below random. Confirmed across four
> architectures, six model families, three bucketings and a 1,728-config grid.

So some confidence has to be collected. How much, and which items, are also
measured:

- **k = 3 is the knee.** Going 0→3 calibration items buys +17 to +33 accuracy
  points; going 3→7 buys +1.4 to +2.6. Three is where the curve flattens.
- **Never use easy items.** Across every strategy tested, the three easiest
  questions were the worst calibration set on both quizzes (−1.9 to −2.7 pts).
  The three shipped here are medium-to-hard syllogism and conditional-logic
  items where the intuitive answer is wrong often enough to spread ratings out.
- **Report probabilities, not hard labels.** At AUC ≈ 0.81 the *ranking* across
  questions is much more trustworthy than any single argmax, which is why the
  chart leads with a line and a band rather than three coloured buckets.

### The honest caveat

Every measured figure above comes from calibration items in the **same subject**
as the predicted questions. This quiz calibrates on general reasoning and
predicts technical questions. Cross-domain transfer was measured directly:
within-subject r = 0.794, cross-subject r = 0.557, and nine questions from
another subject were worth less than three from the same one.

**Expect several points below the reported accuracy.** The results page says so
to the student rather than hiding it. If you want the higher number, replace the
three reasoning items in `src/data/questions.js` with three medium-to-hard items
from the same subject as the technical block — the code needs no other change.

---

## The model

Fitted by `scripts/fit_web_model.py` on 16,990 rated answers from 1,803 students
(1,962 attempts), then exported to `src/model/coefficients.json` as plain weight
vectors that `src/model/predict.js` evaluates in the browser.

Two heads over the same 23 standardised features:

- **Multinomial logistic** → P(Low), P(Medium), P(High), bucketing
  `1-2 / 3-4 / 5`
- **Ridge** → the continuous 1–5 rating, which drives the line on the chart

Validated with 5-fold `GroupKFold` by `student_uid`, 8 random calibration draws,
scored **only on rows whose ratings were withheld**:

| Metric | Value |
|---|---|
| Accuracy | **63.74%** (sd 0.37) |
| Majority-class baseline | 48.73% |
| Lift | **+15.0 points** |
| Balanced accuracy | 0.6340 |
| Macro F1 | 0.5841 |
| Min-class recall | 0.5473 |
| QWK | 0.5127 |
| AUC (OvR macro) | 0.8143 |

Top features by standardised coefficient — the calibration block dominates,
exactly as the scope condition predicts:

```
calib_high_frac  0.599     is_correct    0.260     log_time        0.159
calib_mean       0.479     calib_max     0.259     any_opt_change  0.124
calib_std        0.384     calib_range   0.225     time_rel        0.109
```

### Leave-one-out, in two places

1. **Fitting.** Each row's calibration features are built from the *other*
   ratings in that attempt. Calibration rows exclude their own rating; technical
   rows see all three. `predict.js` reproduces this exactly.
2. **At results time.** Each of the student's three ratings is re-predicted from
   the other two. The mean absolute error across those three is shown on the
   results page as the honest read on whether that student's ratings are
   coherent enough to project forward — and it drives the wording of the caveat
   they see.

> This is deliberately the *correct* LOO, not the buggy one. `transform("std")`
> and friends include the row itself; only `mean` is true LOO. That bug inflated
> an earlier reported figure by ~11 points (`CONTEXT.md` §9.2).

### Parity is enforced, not assumed

Two implementations of the same model will drift. `scripts/parity_test.mjs` runs
a fixed attempt through the browser code; `scripts/parity_check.py` recomputes it
in numpy and diffs every intermediate:

```bash
node scripts/parity_test.mjs > /tmp/js.json
python scripts/parity_check.py /tmp/js.json
# PARITY OK — JS and Python agree to 1e-9 on every value
```

Run this after touching either side.

---

## Auth

**Google sign-in**, via `signInWithPopup` with an automatic fallback to
`signInWithRedirect` — popups are blocked outright in several in-app browsers
(Instagram, LinkedIn, various Android webviews) that students do open links in,
and a silent failure there looks like a broken site.

`prompt: "select_account"` is always set. Students share lab machines, and
silently reusing whichever account the browser last used would file the attempt
under the wrong person.

**Domain allowlist** — optional, via `VITE_ALLOWED_EMAIL_DOMAINS` (comma-
separated, empty = any Google account). It gates the UI and signs out a
mismatched account immediately, but it is *not* a security boundary; a
determined user can call the API directly. `firestore.rules` carries a
commented-out `request.auth.token.email.matches(...)` line to enforce it
server-side as well.

**Retakes are allowed but flagged.** Every attempt carries `attemptNumber` and
`isRetake`, counted from the student's own prior documents via an aggregation
query (one read unit, not one per document). Filter to `attemptNumber == 1` to
keep only first sittings — the thing `clean_and_eda.py` currently has to do the
hard way, after the fact, at a cost of 453 attempts and 3,461 rows.

Nobody gets locked out: a student who loses connection mid-quiz just starts
again, and the extra sitting is visible in the data rather than silently mixed
into it.

## The quiz flow

```
Sign in        Google account
Landing        roll number (pre-filled from the email when it contains one),
               optional CGPA
   │
Section 1      3 reasoning questions · confidence collected · forward-only
   │           (no back-navigation: a revised rating is no longer an
   │            independent read, and the LOO stats would be self-referential)
Section 2      12 technical questions · NO confidence widget
   │           free navigation, flag-for-review, telemetry only
Submit         → scored in-browser → written to Firestore
   │
Results        line chart + band · LOO table · behavioural states · per-question review
```

### Telemetry

Per question: `timeSpent`, `optionChanges`, `markedForReview`,
`reviewClickCount`, `visits`, `isCorrect` — the same field names the existing
`Bis-quiz` app writes, so downstream exports keep working.

Time accrues **only while the tab is visible and focused**. Without that, a
student who alt-tabs away registers as deep deliberation, and time feeds the
model.

### Behavioural state

A deterministic rule over logged fields — no model, no caveat:

| State | Rule |
|---|---|
| Mastery | correct, no hesitation |
| Shaky | correct, hesitated |
| Misconception | wrong, no hesitation |
| Confusion | wrong, hesitated |

*hesitated = changed an option, flagged it, revisited it, or spent longer than
their own attempt average.*

---

## Running locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

Fill in your Firebase web config in `.env.local`, then open http://localhost:5173.

Firebase config is optional for development. Without it the app **skips the
sign-in screen entirely**, runs on the bundled question bank, shows a banner
saying the attempt will not be saved, and everything else — scoring, chart,
analysis — works normally.

`localhost` is in Firebase's authorized-domain list by default, so Google
sign-in works locally as soon as you add real config. The deployed domain is
not — see step 7 of [DEPLOY_AWS.md](DEPLOY_AWS.md).

```bash
npm run build
npm run preview
npm run seed
npm run check
```

`build` writes `dist/`; `preview` serves that build on :4173; `seed` pushes the
question bank to Firestore; `check` reports what is actually in Firestore. The
last two need `serviceAccountKey.json`.

### Refitting the model

```bash
source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate badminton
python scripts/fit_web_model.py      # rewrites src/model/coefficients.json
node scripts/parity_test.mjs > /tmp/js.json && python scripts/parity_check.py /tmp/js.json
```

Reads `../cleaned.pkl`. Takes about two minutes.

---

## Editing the questions

`src/data/questions.js`, then `npm run seed`.

The calibration block **must stay at exactly 3 items** — the model was fitted
for k=3, and `loadBank()` falls back to the bundled bank rather than serve a
different count. The technical block can be any length.

Keep calibration items medium-to-hard. That is the one robust finding on item
selection, and easy items measurably degrade the result.

---

## The chart

`src/components/ConfidenceChart.jsx` — hand-drawn SVG styled after a seaborn
`lineplot(errorbar="sd")`: whitegrid, recessive spines, band under the line.

It is SVG rather than a rendered seaborn PNG because the analysis has to happen
in the student's browser — there is no server to run Python on, which is the
same constraint that made the model a logistic regression. The visual grammar is
seaborn's; the runtime is the browser's.

- **Blue line + markers** — predicted confidence, 1–5
- **Blue band** — ±1 SD of the prediction, from the class distribution; it
  widens exactly where the model is unsure
- **Orange dots** — the three ratings actually given, with a dashed drop to the
  held-out prediction for the same question
- **Shaded left region** — the calibration block

Colours are categorical slots 1–3 of the validated default palette, checked with
the `dataviz` validator in both light and dark modes (worst all-pairs CVD ΔE 9.2
light / 9.4 dark, against a ≥8 target). Aqua sits below 3:1 contrast on the light
surface, so the relief rule applies: a legend is always present and a full table
view of the same numbers sits behind the Chart/Table toggle.

Light, dark and system themes all ship; the toggle is in the top bar.

---

## Data model

```
questions/{id}         block, type, difficulty, text, prompt, options,
                       correctAnswer, explanation, order
settings/config        title, subtitle, totalTimeAllowedMinutes, collectCgpa
quiz_attempts/{auto}   rollNumber, studentCgpa, studentUid, studentEmail,
                       studentName, attemptNumber, isRetake, timestamp,
                       startedAt, calibrationRatings[3], calibrationLooMae,
                       calibrationReliability, correctCount, correctPercentage,
                       behavioralMetrics{ questionId → { …telemetry,
                         predictedConfidence, predictedClass, predictedProbs,
                         predictedSd, behaviouralState } }
```

`firestore.rules` makes attempts write-once and readable only by the student who
created them (which is what the prior-attempt count needs), never updatable or
deletable from the browser, and the bank read-only. Deploy them — the default
production rules deny everything and the default test rules expose every attempt
to the internet.

---

## Files

| Path | Purpose |
|---|---|
| `src/model/predict.js` | LOO stats, feature builder, both heads, state taxonomy |
| `src/model/coefficients.json` | Exported weights + validated metrics |
| `src/components/ConfidenceChart.jsx` | The seaborn-style line plot |
| `src/components/Results.jsx` | Results page: stats, chart, LOO table, review |
| `src/components/CalibrationStage.jsx` | Section 1 — the only place a rating widget exists |
| `src/components/TechnicalStage.jsx` | Section 2 — navigation, flags, telemetry |
| `src/components/SignIn.jsx` | Google sign-in screen, popup → redirect fallback |
| `src/config/firebase.js` | Auth wiring, domain allowlist, Firestore handle |
| `src/lib/telemetry.js` | Focus-aware per-question timing and interaction counts |
| `src/lib/firestore.js` | All reads and writes, with the bundled-bank fallback |
| `scripts/fit_web_model.py` | Fits and exports the model |
| `scripts/parity_test.mjs` · `parity_check.py` | JS↔Python parity harness |
| `scripts/seed_firestore.mjs` | Pushes the bank to Firestore |
| `scripts/check_firestore.mjs` | Read-only health check: bank, attempts, field integrity |
| `scripts/deploy_aws.sh` | Build → S3 → CloudFront invalidation |
| `DEPLOY_AWS.md` | One-time AWS setup, then the one-line deploy |
| `METHODOLOGY.md` | Full methodology: data, all 23 features, validation, limitations |
