# Confidence Calibration Quiz

A quiz that collects confidence ratings on **three** reasoning questions, then
**predicts** confidence on every technical question that follows — and shows the
student a chart of the result.

React + Vite. **Firestore is the only backend for the quiz**; the model runs in
the browser. Deploys to AWS as a static site (S3 + CloudFront) — see
[DEPLOY_AWS.md](DEPLOY_AWS.md).

After scoring, a small Node server (`server/`) sends the attempt to **Gemini via
LangChain** and returns per-question study guidance — see
[Study guidance](#study-guidance-gemini). It is the one part that needs a
server, because it holds the API key.

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

### Study guidance (Gemini)

Every question ends up with two labels: the **behavioural state** above (a rule,
a fact) and the **predicted confidence** class Low / Medium / High (a model
estimate, right about two times in three). `server/guidance.mjs` sends both,
with the question, the options, the student's answer and the explanation, to
Gemini in **one consolidated LangChain call** per attempt:

```
ChatPromptTemplate (system rubric + attempt JSON)
  → ChatGoogleGenerativeAI.withStructuredOutput(zod schema)
  → retried once only if the JSON fails to parse
```

Gemini returns, for each question, a **severity** (`none` / `low` / `moderate` /
`high`), a diagnosis that names the faulty belief behind the option chosen, the
topics tested, and next steps. Topics form one catalog per attempt (a concept
shared by three questions is one topic), and each topic carries **free study
resources** — YouTube videos, blog posts and tutorials, documentation, free
courses and practice sets — scaled to the worst severity on that topic: three
or four for a serious gap, two for a minor one, one to go deeper on mastery. It
also returns an overall summary, the weak concepts with their resources, and a
study plan. The results page shows the overall view in a "What to work on" card
and the per-question advice under each question.

The severity rubric lives in the system prompt. A confident wrong answer
(misconception) is the most severe case; a wrong answer on an easy or medium
item goes up a step; a near-miss on a hard item comes down a step. The prompt
tells Gemini that the predicted confidence is a soft signal, that the student's
own rating overrides it on calibration items, and that it must never present
the estimate as fact.

Safeguards:

- **The key stays on the server.** The browser posts to `/api/guidance`, which
  Vite proxies to `server/index.mjs` on :8787. LangChain is not in the bundle.
- **No personal data is sent.** No roll number, email, name or CGPA; only the
  questions and how they were answered.
- **Every resource link is checked before it is shown** (`server/resources.mjs`).
  Gemini picks good resources but misremembers their addresses: from memory,
  8 of 8 YouTube links it produced were invented video IDs. So YouTube links
  are confirmed through YouTube's oEmbed endpoint (which also supplies the real
  title and channel), pages are fetched and must load a real, on-topic page
  that is not a site's home page, and anything that fails becomes a labelled
  search — YouTube search for "title channel", or a Google search restricted to
  the site. Typically about a third of links verify directly; the rest are
  searches that land on the named resource. Checks take a few seconds, run in
  parallel and are cached for a day. The fetcher only reaches public hosts on
  standard ports and re-checks every redirect, since the URLs are model output.
- **Direct video links need search grounding.** Gemini's Google Search tool
  would return real, current video URLs, but it needs a billed Gemini plan (a
  free-tier key gets a 429), so it is not used.
- **The page never waits on it.** Scores render immediately, guidance fills in
  when it arrives (typically 30–60 s), and any failure shows a readable reason
  with a retry button.
- **One overloaded model does not fail the request.** The chain is tried model
  by model (see `GEMINI_FALLBACK_MODELS` below); a bad key or a cancelled
  request stops at once instead.

To see exactly what Gemini is sent, without taking the quiz:

```bash
npm run guidance:prompt                # formatted prompt for a sample attempt
npm run guidance:prompt -- --schema    # JSON schema Gemini must answer in
npm run guidance:prompt -- --live      # call Gemini and print the result
```

---

## Running locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

Fill in your Firebase web config and `GEMINI_API_KEY` in `.env.local`, then
open http://localhost:5173. `npm run dev` starts both Vite (`web`) and the
guidance server (`api`); `npm run dev:web` starts Vite alone.

Without `GEMINI_API_KEY` the quiz still works end to end; the results page just
says study guidance is unavailable. The server re-reads `.env.local` on every
request, so adding or changing the key needs no restart — press **Try again** on
the results page.

`GEMINI_MODEL` overrides the default `gemini-3.8-flash`. `GEMINI_THINKING_LEVEL`
defaults to `low`, which measured 37 s against 52 s at the model's default on a
15-question attempt with identical severities; set it empty for the default. When a model answers
503 "high demand" (common on the newest Flash) or is not available to the key,
the server moves on to `GEMINI_FALLBACK_MODELS` (default 3.5 → 3.7 → 3.6
Flash, then `gemini-3.5-flash-lite` as a last resort), and the page credits whichever model
actually answered. `curl localhost:8787/api/health` shows the key status and
model order.

Firebase config is optional for development. Without it the app **skips the
sign-in screen entirely**, runs on the bundled question bank, shows a banner
saying the attempt will not be saved, and everything else — scoring, chart,
analysis — works normally.

`localhost` is in Firebase's authorized-domain list by default, so Google
sign-in works locally as soon as you add real config. The deployed domain is
not — see step 7 of [DEPLOY_AWS.md](DEPLOY_AWS.md).

The S3 + CloudFront deployment serves static files only, so the guidance API is
deployed separately, to **AWS Lambda behind a function URL**, by
`scripts/deploy_guidance_lambda.sh` — see step 9 of
[DEPLOY_AWS.md](DEPLOY_AWS.md). Deployed, it only serves students signed in
to the quiz's Firebase project. A production build without
`VITE_GUIDANCE_API_URL` leaves guidance out rather than showing an error.

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
| `server/guidance.mjs` | LangChain prompt, Gemini chain, request/response schemas |
| `server/index.mjs` | Guidance API (`POST /api/guidance`), holds the Gemini key |
| `server/resources.mjs` | Checks every resource link; falls back to labelled searches |
| `server/lambda.mjs` | Same API as a Lambda function-URL handler (production) |
| `server/auth.mjs` | Verifies the student's Firebase ID token on the deployed API |
| `scripts/deploy_guidance_lambda.sh` | Builds and deploys the Lambda; `--print-policy` for IAM |
| `src/lib/guidance.js` | Builds the guidance request from a scored attempt |
| `scripts/preview_guidance_prompt.mjs` | Prints the exact prompt; `--live` calls Gemini |
| `src/model/coefficients.json` | Exported weights + validated metrics |
| `src/components/ConfidenceChart.jsx` | The seaborn-style line plot |
| `src/components/Results.jsx` | Results page: stats, study guidance, chart, LOO table, review |
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
