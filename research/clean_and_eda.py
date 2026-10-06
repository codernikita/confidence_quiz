#!/usr/bin/env python
"""
STAGE 1 — Deep clean + EDA, writing a single analysis-ready dataset.

Fixes every data-integrity defect found in the audit:
  1. dms_quiz_one holds TWO disjoint quiz forms; 507 rows have no bank entry.
  2. Question_Bank cols 155-165 are byte-identical duplicates of 0-10.
  3. Taxonomy not normalised ('Theory'/'theoretical', 'Apply'/'Applying').
  4. Repeat attempts: a student re-taking the SAME subject up to 11 times.
     Only the FIRST attempt per (student, subject) is a clean exam observation;
     later ones are practice//memory-contaminated.
  5. Dead columns (all-null, zero-variance).
  6. confidence==0 conflated with "unanswered".

Outputs: cleaned.pkl  +  eda_report.txt
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from idea1_hierarchical import load_pooled, build_features

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = []
def say(s=""):
    print(s); LOG.append(str(s))


def main():
    say("="*80); say("STAGE 1 — CLEAN + EDA"); say("="*80)

    B, Q, E = load_pooled(verbose=False)
    say(f"\n[raw] behaviour rows={len(B):,} attempts={B.attempt_id.nunique():,} "
        f"students={B.student_uid.nunique():,} questions={B.question_id.nunique()}")
    say(f"[raw] bank questions={len(Q)}  embeddings={len(E)}")

    d, emb_cols = build_features(B, Q, E, verbose=False)
    n0 = len(d)

    # ---- FIX 1: drop rows whose question has no bank entry (the orphan quiz form)
    inbank = d.question_id.isin(set(Q.question_id))
    say(f"\n[clean 1] rows whose question is absent from Question_Bank: "
        f"{(~inbank).sum():,} ({(~inbank).mean():.2%}) -> dropped")
    for src, g in d[~inbank].groupby("__src"):
        say(f"          {src}: {len(g)} rows, {g.attempt_id.nunique()} attempts")
    d = d[inbank].copy()

    # ---- FIX 4: keep only the FIRST attempt per (student, subject)
    d["ts"] = pd.to_datetime(d.timestamp, errors="coerce", utc=True)
    order = (d.groupby(["student_uid", "__src", "attempt_id"]).ts.min()
               .reset_index().sort_values("ts"))
    first = order.groupby(["student_uid", "__src"]).attempt_id.first()
    keep_ids = set(first.values)
    dropped_att = d.attempt_id.nunique() - len(keep_ids)
    say(f"\n[clean 4] repeat attempts on the SAME subject: {dropped_att:,} attempts dropped "
        f"({d[~d.attempt_id.isin(keep_ids)].shape[0]:,} rows)")
    say(f"          rationale: a 2nd-11th sitting of the same quiz is memory-contaminated;")
    say(f"          confidence there reflects recall, not first-encounter understanding.")
    d = d[d.attempt_id.isin(keep_ids)].copy()

    # ---- FIX 6: split the two meanings of confidence==0
    d["is_unanswered"] = d.unanswered
    d["conf_raw"] = d.confidence_rating
    d["conf_valid"] = np.where(d.unanswered, np.nan, np.clip(d.confidence_rating, 1, 5))
    say(f"\n[clean 6] unanswered rows: {int(d.is_unanswered.sum()):,} "
        f"({d.is_unanswered.mean():.2%}) -> confidence set to NaN, kept as separate flag")
    zero_answered = int(((d.confidence_rating == 0) & (~d.unanswered)).sum())
    say(f"          answered-but-rated-0 rows: {zero_answered:,} -> clipped to 1")

    # ---- FIX 5: report dead columns
    dead = [c for c in d.columns if d[c].notna().sum() == 0]
    const = [c for c in d.select_dtypes(include=[np.number]).columns
             if d[c].nunique(dropna=True) <= 1]
    say(f"\n[clean 5] all-null columns: {len(dead)} | zero-variance numeric: {len(const)}")

    # ---- attempt-level frame (for attempt-level targets) ----
    att = d.groupby("attempt_id").agg(
        student=("student_uid", "first"), src=("__src", "first"),
        cgpa=("student_cgpa", "first"),
        n_items=("question_id", "size"),
        n_correct=("is_correct", "sum"),
        acc=("is_correct", "mean"),
        unans_frac=("is_unanswered", "mean"),
        tot_time=("time_spent", "sum"),
        med_time=("time_spent", "median"),
        fast_frac=("time_spent", lambda s: (s < 3).mean()),
        mean_changes=("option_changes", "mean"),
        sum_changes=("option_changes", "sum"),
        mean_review=("review_click_count", "mean"),
        any_review=("marked_for_review", "max"),
        conf_mean=("conf_valid", "mean"),
        conf_sd=("conf_valid", "std"),
        conf_nuniq=("conf_valid", "nunique"),
        conf_mode_share=("conf_raw", lambda s: s.value_counts(normalize=True).iloc[0]),
        time_remaining=("time_remaining_at_submission", "first"),
    )
    cal = {}
    for aid, g in d[~d.is_unanswered].groupby("attempt_id"):
        if g.conf_valid.nunique() > 1 and g.is_correct.nunique() > 1:
            cal[aid] = spearmanr(g.conf_valid, g.is_correct).statistic
    att["calib"] = att.index.map(cal)
    att["careless"] = (att.conf_mode_share.eq(1.0) | att.tot_time.lt(60)
                       | att.unans_frac.ge(0.5))

    say("\n" + "="*80); say("EDA"); say("="*80)
    say(f"\n[final] rows={len(d):,} (from {n0:,}) | attempts={len(att):,} | "
        f"students={d.student_uid.nunique():,} | questions={d.question_id.nunique()} "
        f"| with embeddings={d[d._has_emb].question_id.nunique()}")

    say("\n-- target candidates, ITEM level (answered rows only) --")
    a = d[~d.is_unanswered]
    say(f"   is_correct                 : mean={a.is_correct.mean():.3f}  "
        f"majority-baseline={max(a.is_correct.mean(),1-a.is_correct.mean()):.3f}")
    for lo, name in [(3, "conf>=4 (High)"), (4, "conf==5")]:
        v = (a.conf_valid >= lo + 1).mean() if lo == 3 else (a.conf_valid == 5).mean()
        say(f"   {name:26s} : mean={v:.3f}  majority-baseline={max(v,1-v):.3f}")
    say(f"   3-class B (1-3/4/5)        : " + " ".join(
        f"{c}={v:.3f}" for c, v in zip(["Low","Med","High"],
        np.bincount(np.where(a.conf_valid<=3,0,np.where(a.conf_valid==4,1,2)),minlength=3)/len(a))))

    say("\n-- target candidates, ATTEMPT level (noise averages out over 13 items) --")
    for thr in [0.4, 0.5, 0.6]:
        v = (att.acc >= thr).mean()
        say(f"   passed >= {int(thr*100)}%            : mean={v:.3f}  "
            f"majority-baseline={max(v,1-v):.3f}")
    say(f"   careless attempt           : mean={att.careless.mean():.3f}  "
        f"majority-baseline={max(att.careless.mean(),1-att.careless.mean()):.3f}")
    say(f"   high mean confidence (>=4) : mean={(att.conf_mean>=4).mean():.3f}  "
        f"majority-baseline={max((att.conf_mean>=4).mean(),1-(att.conf_mean>=4).mean()):.3f}")

    say("\n-- correlations with is_correct (answered rows) --")
    num = ["time_spent","time_z_within_q","option_changes","review_click_count",
           "marked_for_review","student_cgpa","conf_valid","att_unans_frac",
           "att_log_total_time","opt_set_cohesion","opt_nearest_trap"]
    cc = a[num+["is_correct"]].corr(numeric_only=True)["is_correct"].drop("is_correct")
    say(cc.sort_values(key=abs, ascending=False).round(3).to_string())

    say("\n-- correlations with attempt accuracy --")
    numa = ["cgpa","unans_frac","tot_time","med_time","fast_frac","mean_changes",
            "mean_review","conf_mean","conf_sd","time_remaining"]
    cca = att[numa+["acc"]].corr(numeric_only=True)["acc"].drop("acc")
    say(cca.sort_values(key=abs, ascending=False).round(3).to_string())

    say("\n-- per-question difficulty spread (drives how learnable is_correct is) --")
    qs = a.groupby("question_id").is_correct.agg(["mean","size"])
    say(f"   questions={len(qs)}  acc range {qs['mean'].min():.3f}-{qs['mean'].max():.3f}  "
        f"sd={qs['mean'].std():.3f}  median n per q={int(qs['size'].median())}")

    d.to_pickle(os.path.join(OUT, "cleaned.pkl"))
    att.to_pickle(os.path.join(OUT, "cleaned_attempts.pkl"))
    with open(os.path.join(OUT, "eda_report.txt"), "w") as f:
        f.write("\n".join(LOG))
    say(f"\nSaved -> cleaned.pkl ({len(d):,} rows), cleaned_attempts.pkl "
        f"({len(att):,} attempts), eda_report.txt")


if __name__ == "__main__":
    main()
