#!/usr/bin/env python
"""
Idea 1 — Hierarchical Bayesian Ordinal Model with Careless-Response Mixture
==========================================================================

Predicts student confidence (0-5) on quiz items, with four stacked components:

  1. HURDLE          confidence==0 is mostly "did not answer", not "very unconfident".
                     Stage 1 models P(answered); stage 2 models confidence 1-5 | answered.
  2. ORDINAL         Cumulative-link (proportional-odds) model. Respects rung ordering
                     without assuming equal spacing between rungs.
  3. PARTIAL POOLING Crossed random intercepts for student and question. Students/questions
                     with little data shrink toward the population mean.
                       - alpha_s ~ N(mu + w*cgpa_s, sigma_a^2)       <- captures the 53%
                       - beta_q  ~ N(W*PCA8(emb_q) + V*meta_q, sigma_b^2)  <- the 7%
                     The embedding touches the model ONCE, as a prior on 92-126 item
                     intercepts -- never as a per-row feature. This is what prevents the
                     memorisation failure (SBERT-alone rho: 0.225 random -> 0.049 grouped).
  4. MIXTURE         Each attempt is a latent blend of "engaged" and "careless" regimes.
                     Marginalised in closed form (no EM). Time effects are fit INSIDE the
                     engaged regime, so "fast" can mean mastery there and abandonment
                     elsewhere (measured: 87% vs 15% accuracy at 3-5s).

Inference: mean-field stochastic variational inference (SVI) in PyTorch.
Gives posterior means AND standard deviations. No numpyro/JAX required.

IMPORTANT — leakage discipline:
  * PCA of embeddings, LOO difficulty encoding, and all normalisation are fit
    INSIDE the training fold only.
  * The engagement mixture is parameterised by NON-LABEL attempt features only
    (unanswered fraction, timing patterns). It never sees confidence ratings, so
    straightlining is discovered unsupervised rather than handed to the model.
  * Evaluation subsets ("clean" vs "careless") are defined post-hoc for reporting
    and are never model inputs.

Usage:
    python idea1_hierarchical.py                  # all three CV schemes
    python idea1_hierarchical.py --quick          # fewer epochs, smoke test
    python idea1_hierarchical.py --splits question student
    python idea1_hierarchical.py --epochs 3000 --seeds 0 1 2
"""

import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score, cohen_kappa_score, brier_score_loss,
)
from sklearn.model_selection import GroupKFold, KFold

DATA_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "compiled_files", "*", "*.xlsx")
EMB_GLOB_TMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "compiled_files", "*", "*%s_embeddings.csv")


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------

def load_pooled(embedding="SentenceBERT", verbose=True):
    """Pool all workbooks. Returns (behaviour_df, bank_df, embedding_df)."""
    beh, bank = [], []
    for f in sorted(glob.glob(DATA_GLOB)):
        tag = os.path.basename(f).replace(".xlsx", "")
        try:
            xl = pd.ExcelFile(f)
        except Exception as e:
            print(f"  ! skipping {tag}: {e}")
            continue
        if "ML_Behavioral_Data" in xl.sheet_names:
            b = xl.parse("ML_Behavioral_Data"); b["__src"] = tag; beh.append(b)
        if "Question_Bank" in xl.sheet_names:
            q = xl.parse("Question_Bank"); q["__src"] = tag; bank.append(q)
    B = pd.concat(beh, ignore_index=True)
    Q = pd.concat(bank, ignore_index=True)

    # Question_Bank cols 155-165 are byte-identical duplicates of 0-10 -> drop
    Q = Q.loc[:, ~Q.columns.str.endswith(".1")]
    Q = Q.drop_duplicates("question_id")

    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(EMB_GLOB_TMPL % embedding))],
                  ignore_index=True).drop_duplicates("question_id")

    if verbose:
        print(f"  pooled behaviour : {len(B):,} rows | {B.attempt_id.nunique():,} attempts "
              f"| {B.student_uid.nunique():,} students | {B.question_id.nunique()} questions")
        print(f"  pooled bank      : {len(Q)} questions ({Q.question_text.nunique()} unique texts)")
        print(f"  {embedding} embeddings : {len(E)} questions x "
              f"{len([c for c in E.columns if c != 'question_id'])} dims")
    return B, Q, E


def normalise_taxonomy(s):
    """'Applying'->'apply', 'Theory'/'theoretical'->'theory', etc."""
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r"(ing|ical)$", "", regex=True)
    return s.replace({"theoret": "theory", "numer": "numerical"})


def build_features(B, Q, E, verbose=True):
    """Construct every feature block. Returns one row per behavioural record."""
    d = B.copy()
    d["unanswered"] = d.final_selected_option.astype(str).str.strip().isin(["None", "nan", ""])
    d["answered"] = (~d.unanswered).astype(float)

    # ---- attempt-level features (NO confidence labels -> safe as model inputs) ----
    agg = d.groupby("attempt_id").agg(
        att_n_items=("question_id", "size"),
        att_unans_frac=("unanswered", "mean"),
        att_total_time=("time_spent", "sum"),
        att_med_time=("time_spent", "median"),
        att_fast_frac=("time_spent", lambda s: (s < 3).mean()),
        att_mean_changes=("option_changes", "mean"),
    )
    agg["att_log_total_time"] = np.log1p(agg.att_total_time)
    agg["att_log_med_time"] = np.log1p(agg.att_med_time)
    d = d.join(agg, on="attempt_id")

    # ---- post-hoc "careless" flag: EVALUATION ONLY, never a model input ----
    # (uses confidence straightlining, which the model must not see)
    ev = d.groupby("attempt_id").agg(
        conf_mode_share=("confidence_rating", lambda s: s.value_counts(normalize=True).iloc[0]),
        tt=("time_spent", "sum"), uf=("unanswered", "mean"))
    ev["eval_careless"] = (ev.conf_mode_share.eq(1.0) | ev.tt.lt(60) | ev.uf.ge(0.5))
    d = d.join(ev[["eval_careless"]], on="attempt_id")

    # ---- question metadata ----
    qm = Q.set_index("question_id")
    for col, default in [("type", "unknown"), ("blooms_level", "unknown"), ("difficulty", "unknown")]:
        if col in qm.columns:
            d["q_" + col] = normalise_taxonomy(d.question_id.map(qm[col])).fillna(default)
        else:
            d["q_" + col] = default
    read_cols = ["flesch_ease", "gunning_fog_index", "lexical_diversity",
                 "feat_n_words", "feat_parse_tree_depth", "feat_avg_dependency_length",
                 "feat_dale_chall", "feat_n_clauses"]
    for c in read_cols:
        d["q_" + c] = pd.to_numeric(d.question_id.map(qm[c]), errors="coerce") if c in qm.columns else np.nan
    d["correct_answer"] = d.question_id.map(qm["correct_answer"]).astype(str) if "correct_answer" in qm.columns else ""

    # ---- option-set reconstruction + semantic similarity features ----
    # (options are recoverable from observed selections; no option embedding file exists)
    opts = (d[~d.unanswered].groupby("question_id").final_selected_option
            .apply(lambda s: sorted(set(s.astype(str)))))
    all_txt = sorted(set(x for v in opts for x in v) | set(d.correct_answer.astype(str)))
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    tv.fit(all_txt)
    vecs = {t: tv.transform([t]).toarray()[0] for t in all_txt}

    def cos(a, b):
        va, vb = vecs.get(str(a)), vecs.get(str(b))
        if va is None or vb is None:
            return np.nan
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else np.nan

    cohesion, trap = {}, {}
    for qid, o in opts.items():
        if len(o) < 2:
            cohesion[qid], trap[qid] = np.nan, np.nan
            continue
        V = np.vstack([vecs[t] for t in o])
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = Vn @ Vn.T
        cohesion[qid] = (S.sum() - len(o)) / (len(o) * (len(o) - 1))
        ca = str(qm["correct_answer"].get(qid, "")) if "correct_answer" in qm.columns else ""
        sims = [cos(ca, t) for t in o if str(t) != ca]
        sims = [s for s in sims if not np.isnan(s)]
        trap[qid] = max(sims) if sims else np.nan

    d["opt_set_cohesion"] = d.question_id.map(cohesion)
    d["opt_nearest_trap"] = d.question_id.map(trap)
    d["opt_n_options"] = d.question_id.map({k: len(v) for k, v in opts.items()})
    d["opt_sim_chosen_correct"] = [
        cos(a, b) if not u else np.nan
        for a, b, u in zip(d.final_selected_option, d.correct_answer, d.unanswered)
    ]

    # ---- response-level ----
    d["time_z_within_q"] = d.groupby("question_id").time_spent.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    d["time_rel_student"] = d.time_spent / (d.groupby("student_uid").time_spent.transform("median") + 1.0)
    d["log_time"] = np.log1p(d.time_spent)

    # ---- embeddings kept separate; PCA is fit per-fold ----
    emb_cols = [c for c in E.columns if c != "question_id"]
    d = d.merge(E, on="question_id", how="left")
    d["_has_emb"] = d[emb_cols[0]].notna()

    if verbose:
        print(f"  rows with question embeddings: {d._has_emb.sum():,} / {len(d):,}")
        print(f"  unanswered: {d.unanswered.mean():.1%} | "
              f"eval-careless attempts: {ev.eval_careless.mean():.1%}")
    return d.reset_index(drop=True), emb_cols



def apply_tier_filter(d, tier):
    """Restrict to an engagement tier (cumulative). Returns filtered frame.

    T1  not careless   : not straightlined, not <60s total, not >=50% blank
    T2  T1 + used >=3 distinct confidence values + <20% blank
    T3  T2 + >=5 minutes total on the attempt

    NOTE: T2/T3 are strict subsets of T1, i.e. of the NON-careless attempts.
    Inside them every row has eval_careless == False, so the careless mixture
    and the careless down-weighting have nothing left to act on, and the
    hurdle stage sees almost no unanswered rows. That is expected, and is
    reported explicitly rather than hidden.
    """
    if tier in (None, "none"):
        return d
    from scipy.stats import spearmanr as _sp
    att = d.groupby("attempt_id").agg(
        unans_frac=("unanswered", "mean"), tot_time=("time_spent", "sum"),
        cms=("confidence_rating", lambda s: s.value_counts(normalize=True).iloc[0]),
        cnu=("confidence_rating", "nunique"))
    keep = ~(att.cms.eq(1.0) | att.tot_time.lt(60) | att.unans_frac.ge(0.5))   # T1
    if tier in ("T2", "T3"):
        keep = keep & att.cnu.ge(3) & att.unans_frac.lt(0.2)
    if tier == "T3":
        keep = keep & att.tot_time.ge(300)
    ids = set(att.index[keep.fillna(False)])
    out = d[d.attempt_id.isin(ids)].reset_index(drop=True)
    print(f"  tier {tier}: kept {out.attempt_id.nunique():,} attempts | "
          f"{out.student_uid.nunique():,} students | {len(out):,} rows | "
          f"{out.question_id.nunique()} questions "
          f"({out[out._has_emb].question_id.nunique()} with embeddings)")
    print(f"    unanswered rows remaining: {int(out.unanswered.sum())} "
          f"({out.unanswered.mean():.2%})  |  careless rows remaining: "
          f"{int(out.eval_careless.sum())} ({out.eval_careless.mean():.2%})")
    return out


# ----------------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------------

class HierarchicalOrdinalMixture(nn.Module):
    """
    Stage 1 (hurdle):   P(answered)     = sigmoid(a_s^r + b_q^r + g_r . x)
    Stage 2 (ordinal):  P(conf=k | ans) = mixture over latent engagement
        engaged : OrderedLogit(eta ; kappa_1..kappa_4)
        careless: Categorical(psi)                       [global response style]
        eta = alpha_s + beta_q + gamma.x + delta.(time_z * pi_engaged) + lambda.opt_feats

    Random effects use the non-centred parameterisation
        alpha_s = mu_a + w_cgpa*cgpa_s + sigma_a * z_s ,  z_s ~ N(0,1)
    so the N(0,1) prior on z is exactly the shrinkage that produces partial pooling.
    Variational posteriors q(z) = N(m, s^2) are learned (mean-field SVI).
    """

    def __init__(self, n_students, n_questions, n_fixed, n_qprior, n_attfeat,
                 n_hurdle_fixed, K=5):
        super().__init__()
        self.K = K
        self.n_students, self.n_questions = n_students, n_questions

        # --- variational params for student / question random effects (stage 2) ---
        self.z_s_mu = nn.Parameter(torch.zeros(n_students))
        self.z_s_logs = nn.Parameter(torch.full((n_students,), -2.0))
        self.z_q_mu = nn.Parameter(torch.zeros(n_questions))
        self.z_q_logs = nn.Parameter(torch.full((n_questions,), -2.0))

        # --- hierarchical scale + prior-mean regressions ---
        self.log_sigma_a = nn.Parameter(torch.tensor(-0.7))   # student SD
        self.log_sigma_b = nn.Parameter(torch.tensor(-0.7))   # question SD
        self.mu_a = nn.Parameter(torch.zeros(1))
        self.w_cgpa = nn.Parameter(torch.zeros(1))
        self.W_qprior = nn.Linear(n_qprior, 1)                # PCA8(emb) + meta -> beta_q prior mean

        # --- fixed effects ---
        self.gamma = nn.Linear(n_fixed, 1, bias=False)
        self.delta_time = nn.Parameter(torch.zeros(1))        # time_z, gated by engagement

        # --- ordinal cutpoints (monotone by construction) ---
        self.kappa_1 = nn.Parameter(torch.tensor([-2.0]))
        self.kappa_d = nn.Parameter(torch.zeros(K - 2))       # softplus increments

        # --- careless component: global categorical response style ---
        self.psi_logits = nn.Parameter(torch.zeros(K))

        # --- engagement mixture weight from NON-LABEL attempt features ---
        self.engage = nn.Linear(n_attfeat, 1)

        # --- hurdle stage ---
        self.z_s_h_mu = nn.Parameter(torch.zeros(n_students))
        self.z_q_h_mu = nn.Parameter(torch.zeros(n_questions))
        self.log_sigma_ah = nn.Parameter(torch.tensor(-0.7))
        self.log_sigma_bh = nn.Parameter(torch.tensor(-0.7))
        self.hurdle_fixed = nn.Linear(n_hurdle_fixed, 1)

    # -- cutpoints ------------------------------------------------------------
    def cutpoints(self):
        return torch.cat([self.kappa_1, self.kappa_1 + torch.cumsum(F.softplus(self.kappa_d), 0)])

    # -- random effects -------------------------------------------------------
    def alpha(self, s_idx, cgpa, sample=True, z_override=None):
        z = self.z_s_mu[s_idx] if z_override is None else z_override
        if sample and self.training and z_override is None:
            z = z + torch.randn_like(z) * self.z_s_logs[s_idx].exp()
        return self.mu_a + self.w_cgpa * cgpa + self.log_sigma_a.exp() * z

    def beta(self, q_idx, qprior, sample=True, z_override=None):
        z = self.z_q_mu[q_idx] if z_override is None else z_override
        if sample and self.training and z_override is None:
            z = z + torch.randn_like(z) * self.z_q_logs[q_idx].exp()
        return self.W_qprior(qprior).squeeze(-1) + self.log_sigma_b.exp() * z

    def engagement(self, att_feats):
        return torch.sigmoid(self.engage(att_feats)).squeeze(-1)

    # -- stage 1 --------------------------------------------------------------
    def hurdle_logit(self, s_idx, q_idx, xh):
        return (self.log_sigma_ah.exp() * self.z_s_h_mu[s_idx]
                + self.log_sigma_bh.exp() * self.z_q_h_mu[q_idx]
                + self.hurdle_fixed(xh).squeeze(-1))

    # -- stage 2 --------------------------------------------------------------
    def eta(self, s_idx, q_idx, cgpa, qprior, x, time_z, pi_eng, sample=True,
            z_s_over=None, z_q_over=None):
        return (self.alpha(s_idx, cgpa, sample, z_s_over)
                + self.beta(q_idx, qprior, sample, z_q_over)
                + self.gamma(x).squeeze(-1) + self.delta_time * time_z * pi_eng)

    @torch.no_grad()
    def predict_marginal(self, s_idx, q_idx, cgpa, qprior, x, time_z, att,
                         seen_s, seen_q, n_mc=64):
        """
        Posterior-predictive confidence probabilities.

        For a student/question SEEN in training we plug in the fitted posterior mean.
        For an UNSEEN one we must integrate over its prior z ~ N(0,1) rather than
        substituting z=0 -- the ordinal link is nonlinear, so plugging in the mean
        of a random effect is NOT the mean of the prediction. Substituting zero
        collapses every unseen-student prediction toward the population mode and
        destroys the low-confidence classes. Monte-Carlo marginalisation fixes it.
        """
        pi = self.engagement(att)
        acc = None
        for _ in range(n_mc):
            zs = torch.where(seen_s, self.z_s_mu[s_idx], torch.randn_like(cgpa))
            zq = torch.where(seen_q, self.z_q_mu[q_idx], torch.randn_like(cgpa))
            eta = self.eta(s_idx, q_idx, cgpa, qprior, x, time_z, pi,
                           sample=False, z_s_over=zs, z_q_over=zq)
            p = self.mixture_logprob(eta, pi).exp()
            acc = p if acc is None else acc + p
        p = acc / n_mc
        return p / p.sum(1, keepdim=True), pi

    @torch.no_grad()
    def predict_hurdle_marginal(self, s_idx, q_idx, xh, seen_s, seen_q, n_mc=64):
        acc = None
        for _ in range(n_mc):
            zs = torch.where(seen_s, self.z_s_h_mu[s_idx], torch.randn_like(xh[:, 0]))
            zq = torch.where(seen_q, self.z_q_h_mu[q_idx], torch.randn_like(xh[:, 0]))
            lg = (self.log_sigma_ah.exp() * zs + self.log_sigma_bh.exp() * zq
                  + self.hurdle_fixed(xh).squeeze(-1))
            p = torch.sigmoid(lg)
            acc = p if acc is None else acc + p
        return acc / n_mc

    def ordinal_logprob(self, eta):
        """log P(conf=k | eta) for k=1..K -> (N, K)."""
        k = self.cutpoints().unsqueeze(0)                     # (1, K-1)
        cdf = torch.sigmoid(k - eta.unsqueeze(1))             # (N, K-1)
        one = torch.ones_like(cdf[:, :1]); zero = torch.zeros_like(cdf[:, :1])
        cdf_full = torch.cat([zero, cdf, one], dim=1)
        probs = (cdf_full[:, 1:] - cdf_full[:, :-1]).clamp_min(1e-9)
        return probs.log()

    def mixture_logprob(self, eta, pi_eng):
        """log[ pi*P_engaged(k) + (1-pi)*psi_k ] -> (N, K), stable via logsumexp."""
        lp_eng = self.ordinal_logprob(eta)
        lp_car = F.log_softmax(self.psi_logits, dim=0).unsqueeze(0).expand_as(lp_eng)
        pe = pi_eng.clamp(1e-6, 1 - 1e-6).unsqueeze(1)
        return torch.logsumexp(torch.stack([pe.log() + lp_eng, (1 - pe).log() + lp_car]), dim=0)

    # -- KL to N(0,1) priors --------------------------------------------------
    def kl(self):
        def _kl(mu, logs):
            s2 = (2 * logs).exp()
            return 0.5 * (s2 + mu ** 2 - 1 - 2 * logs).sum()
        return _kl(self.z_s_mu, self.z_s_logs) + _kl(self.z_q_mu, self.z_q_logs) \
            + 0.5 * (self.z_s_h_mu ** 2).sum() + 0.5 * (self.z_q_h_mu ** 2).sum()


# ----------------------------------------------------------------------------
# 3. FOLD-SAFE DESIGN MATRIX
# ----------------------------------------------------------------------------

FIXED = ["opt_sim_chosen_correct", "opt_set_cohesion", "opt_nearest_trap",
         "option_changes", "review_click_count", "marked_for_review",
         "time_rel_student", "q_loo_difficulty"]
ATT = ["att_unans_frac", "att_log_total_time", "att_fast_frac",
       "att_mean_changes", "att_log_med_time"]
HURDLE_X = ["att_unans_frac", "att_log_total_time", "att_fast_frac",
            "student_cgpa", "q_loo_difficulty"]
QMETA = ["q_flesch_ease", "q_gunning_fog_index", "q_feat_n_words",
         "q_feat_parse_tree_depth", "q_feat_avg_dependency_length"]


def prepare_fold(d, emb_cols, tr, te, n_pca=8):
    """Fit PCA / LOO-difficulty / scalers on TRAIN ONLY, then transform both sides."""
    out = {}

    # -- LOO empirical difficulty (train-fold only) --
    dtr = d.iloc[tr]
    gmean = dtr.is_correct.mean()
    qdiff = dtr.groupby("question_id").is_correct.mean()
    d = d.copy()
    d["q_loo_difficulty"] = d.question_id.map(qdiff).fillna(gmean)

    # -- PCA of embeddings, fit on train-fold questions only --
    has = d._has_emb.values
    Efull = d[emb_cols].values.astype(np.float64)
    Efull = np.nan_to_num(Efull)
    tr_emb_rows = [i for i in tr if has[i]]
    pca = PCA(n_components=min(n_pca, len(set(d.question_id.values[tr_emb_rows])) - 1),
              random_state=0)
    pca.fit(Efull[tr_emb_rows])
    P = np.zeros((len(d), pca.n_components_))
    P[has] = pca.transform(Efull[has])

    # -- question-prior block = PCA + normalised metadata --
    meta = d[QMETA].values.astype(np.float64)
    mmu, msd = np.nanmean(meta[tr], 0), np.nanstd(meta[tr], 0) + 1e-9
    meta = np.nan_to_num((meta - mmu) / msd)
    QPRIOR = np.hstack([P, meta, np.ones((len(d), 1))])

    # -- fixed / attempt / hurdle blocks, scaled on train fold --
    def scale(cols):
        X = d[cols].values.astype(np.float64)
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
        return np.nan_to_num((X - mu) / sd)

    out["X"] = scale(FIXED); out["ATT"] = scale(ATT); out["XH"] = scale(HURDLE_X)
    out["QPRIOR"] = QPRIOR
    out["time_z"] = np.nan_to_num(d.time_z_within_q.values)
    cg = d.student_cgpa.values.astype(np.float64)
    out["cgpa"] = np.nan_to_num((cg - np.nanmean(cg[tr])) / (np.nanstd(cg[tr]) + 1e-9))
    return out


# ----------------------------------------------------------------------------
# 4. TRAIN / PREDICT
# ----------------------------------------------------------------------------

def fit_predict(d, emb_cols, tr, te, epochs=1500, lr=0.05, seed=0, verbose=False,
                careless_weight=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    M = prepare_fold(d, emb_cols, tr, te)

    s_codes = pd.Categorical(d.student_uid).codes.astype(np.int64)
    q_codes = pd.Categorical(d.question_id).codes.astype(np.int64)
    n_s, n_q = s_codes.max() + 1, q_codes.max() + 1

    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    X, ATTf, XH, QP = T(M["X"]), T(M["ATT"]), T(M["XH"]), T(M["QPRIOR"])
    tz, cg = T(M["time_z"]), T(M["cgpa"])
    si, qi = T(s_codes, torch.long), T(q_codes, torch.long)
    ans = T(d.answered.values)
    conf = T(d.confidence_rating.values, torch.long)

    model = HierarchicalOrdinalMixture(n_s, n_q, X.shape[1], QP.shape[1],
                                       ATTf.shape[1], XH.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_t = T(tr, torch.long)
    tr_ans = tr_t[ans[tr_t] > 0]                       # answered rows in train fold
    n_tr = len(tr_t)

    # Careless down-weighting in the ORDINAL loss only (training-time labels are
    # available, so this is legitimate; the flag is never a test-time feature).
    # Without it, straightlined attempts -- where between-student variance is 93.1%
    # vs 36.6% on clean rows -- inflate sigma_student, and marginalising that
    # inflated variance is what destroys cold-student precision.
    w_ord = torch.where(T(d.eval_careless.values.astype(np.float32))[tr_ans] > 0,
                        torch.tensor(careless_weight), torch.tensor(1.0))
    w_ord = w_ord / w_ord.mean()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        # stage 1 — hurdle on ALL train rows
        hl = model.hurdle_logit(si[tr_t], qi[tr_t], XH[tr_t])
        loss_h = F.binary_cross_entropy_with_logits(hl, ans[tr_t])
        # stage 2 — ordinal mixture on ANSWERED train rows
        pe = model.engagement(ATTf[tr_ans])
        eta = model.eta(si[tr_ans], qi[tr_ans], cg[tr_ans], QP[tr_ans],
                        X[tr_ans], tz[tr_ans], pe)
        lp = model.mixture_logprob(eta, pe)
        y = (conf[tr_ans] - 1).clamp(0, model.K - 1)
        loss_o = -(lp.gather(1, y.unsqueeze(1)).squeeze(1) * w_ord).mean()
        loss = loss_h + loss_o + model.kl() / max(n_tr, 1)
        loss.backward(); opt.step()
        if verbose and ep % 300 == 0:
            print(f"      ep {ep:4d}  loss {loss.item():.4f}")

    # ---- predict (marginalising unseen random effects over their prior) ----
    model.eval()
    te_t = T(te, torch.long)
    seen_s = torch.zeros(n_s, dtype=torch.bool); seen_s[si[tr_t].unique()] = True
    seen_q = torch.zeros(n_q, dtype=torch.bool); seen_q[qi[tr_t].unique()] = True
    ss, sq = seen_s[si[te_t]], seen_q[qi[te_t]]
    with torch.no_grad():
        p_ans = model.predict_hurdle_marginal(si[te_t], qi[te_t], XH[te_t], ss, sq).numpy()
        probs_t, pe = model.predict_marginal(si[te_t], qi[te_t], cg[te_t], QP[te_t],
                                             X[te_t], tz[te_t], ATTf[te_t], ss, sq)
        probs = probs_t.numpy(); engage = pe.numpy()
        sigma_a = model.log_sigma_a.exp().item()
        sigma_b = model.log_sigma_b.exp().item()
    return dict(p_answered=p_ans, conf_probs=probs, engagement=engage,
                sigma_a=sigma_a, sigma_b=sigma_b)


# ----------------------------------------------------------------------------
# 5. METRICS
# ----------------------------------------------------------------------------

def to3(y):
    """5-rung ladder -> Low(1-2)/Med(3)/High(4-5). Validated: 57% / 70% / 85% accuracy."""
    return np.where(y <= 2, 0, np.where(y == 3, 1, 2))


def prf_block(y_true, y_pred, labels, names, prefix):
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    rows = []
    for i, nm in enumerate(names):
        rows.append(dict(cls=nm, precision=p[i], recall=r[i], f1=f[i], support=int(sup[i])))
    out = {f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
           f"{prefix}_precision_macro": pm, f"{prefix}_recall_macro": rm, f"{prefix}_f1_macro": fm,
           f"{prefix}_precision_weighted": pw, f"{prefix}_recall_weighted": rw,
           f"{prefix}_f1_weighted": fw}
    return out, pd.DataFrame(rows), confusion_matrix(y_true, y_pred, labels=labels)


def evaluate(d, oof, mask_name, mask, tag):
    """Full metric battery on a row subset."""
    res, tables = {}, {}
    ans_true = d.answered.values.astype(int)
    p_ans = oof["p_answered"]
    probs = oof["conf_probs"]
    conf_true = d.confidence_rating.values

    # ---------- stage 1: hurdle (answered vs unanswered) ----------
    m = mask & ~np.isnan(p_ans)
    if m.sum() > 0 and len(np.unique(ans_true[m])) > 1:
        yhat = (p_ans[m] >= 0.5).astype(int)
        o, tbl, cm = prf_block(ans_true[m], yhat, [0, 1], ["unanswered", "answered"], "hurdle")
        o["hurdle_roc_auc"] = roc_auc_score(ans_true[m], p_ans[m])
        o["hurdle_pr_auc"] = average_precision_score(ans_true[m], p_ans[m])
        o["hurdle_brier"] = brier_score_loss(ans_true[m], p_ans[m])
        res.update(o); tables["hurdle_per_class"] = tbl; tables["hurdle_cm"] = cm

    # ---------- stage 2: confidence 1-5 (answered rows only) ----------
    m2 = mask & (ans_true == 1) & ~np.isnan(probs).any(1)
    if m2.sum() > 0:
        P = probs[m2]
        yt = conf_true[m2]
        yt = np.clip(yt, 1, 5)
        exp_val = (P * np.arange(1, 6)).sum(1)              # expected rung
        argmax = P.argmax(1) + 1

        res["n_eval"] = int(m2.sum())
        res["mae"] = mean_absolute_error(yt, exp_val)
        res["rmse"] = float(np.sqrt(mean_squared_error(yt, exp_val)))
        res["spearman"] = float(spearmanr(yt, exp_val).statistic)
        res["pearson"] = float(pearsonr(yt, exp_val)[0])
        res["adjacent_acc"] = float((np.abs(np.round(exp_val) - yt) <= 1).mean())
        res["qwk"] = float(cohen_kappa_score(yt, argmax, weights="quadratic"))
        # multiclass Brier + ECE
        onehot = np.zeros_like(P); onehot[np.arange(len(yt)), yt - 1] = 1
        res["brier_multiclass"] = float(((P - onehot) ** 2).sum(1).mean())
        cmax = P.max(1); correct = (argmax == yt).astype(float)
        bins = np.linspace(0, 1, 11); ece = 0.0
        for i in range(10):
            b = (cmax >= bins[i]) & (cmax < bins[i + 1])
            if b.sum(): ece += b.mean() * abs(correct[b].mean() - cmax[b].mean())
        res["ece"] = float(ece)

        # ---------- decision rules ----------
        # On a weak-signal, heavily imbalanced ordinal target the MAP/argmax rule
        # collapses to the majority rung, making macro-F1 degenerate. We therefore
        # report three rules. "bal" divides by the model's OWN mean predicted
        # probability (a label-free prior correction), which is the standard fix
        # and is what makes per-class recall interpretable.
        prior = P.mean(0, keepdims=True).clip(1e-9)
        rules = {
            "map": argmax,                                             # argmax p(k)
            "ev":  np.clip(np.round(exp_val), 1, 5).astype(int),       # rounded expected rung
            "bal": (P / prior).argmax(1) + 1,                          # prior-corrected
        }
        for rn, yp in rules.items():
            pre = "conf5" if rn == "map" else f"conf5_{rn}"
            o5, t5, cm5 = prf_block(yt, yp, [1, 2, 3, 4, 5],
                                    ["conf=1", "conf=2", "conf=3", "conf=4", "conf=5"], pre)
            res.update(o5); tables[f"{pre}_per_class"] = t5; tables[f"{pre}_cm"] = cm5

        # ---------- 3-class Low / Medium / High ----------
        P3 = np.stack([P[:, :2].sum(1), P[:, 2], P[:, 3:].sum(1)], 1)
        y3t = to3(yt)
        prior3 = P3.mean(0, keepdims=True).clip(1e-9)
        rules3 = {"map": P3.argmax(1),
                  "ev": to3(np.clip(np.round(exp_val), 1, 5).astype(int)),
                  "bal": (P3 / prior3).argmax(1)}
        for rn, yp in rules3.items():
            pre = "conf3" if rn == "map" else f"conf3_{rn}"
            o3, t3, cm3 = prf_block(y3t, yp, [0, 1, 2],
                                    ["Low(1-2)", "Med(3)", "High(4-5)"], pre)
            res.update(o3); tables[f"{pre}_per_class"] = t3; tables[f"{pre}_cm"] = cm3
        res["conf3_balanced_acc"] = float(np.mean([
            (rules3["bal"][y3t == c] == c).mean() for c in [0, 1, 2] if (y3t == c).sum()]))

        # ---------- baselines ----------
        res["baseline_mean_mae"] = mean_absolute_error(yt, np.full(len(yt), yt.mean()))
        mode = int(pd.Series(yt).mode().iloc[0])
        res["baseline_mode_acc"] = float((yt == mode).mean())
        res["baseline_mode_f1_macro"] = precision_recall_fscore_support(
            yt, np.full(len(yt), mode), average="macro", zero_division=0)[2]
        res["lift_mae_vs_mean"] = (res["baseline_mean_mae"] - res["mae"]) / res["baseline_mean_mae"]

    # ---------- engagement: does the unsupervised mixture rediscover carelessness? ----------
    eng = oof["engagement"]
    m3 = mask & ~np.isnan(eng)
    yt_eng = (~d.eval_careless.values).astype(int)
    if m3.sum() > 0 and len(np.unique(yt_eng[m3])) > 1:
        yhat = (eng[m3] >= 0.5).astype(int)
        o, tbl, cm = prf_block(yt_eng[m3], yhat, [0, 1], ["careless", "engaged"], "engage")
        o["engage_roc_auc"] = roc_auc_score(yt_eng[m3], eng[m3])
        res.update(o); tables["engage_per_class"] = tbl; tables["engage_cm"] = cm

    return res, tables


# ----------------------------------------------------------------------------
# 6. CV DRIVER
# ----------------------------------------------------------------------------

def run_split(d, emb_cols, split, epochs, seed, verbose, careless_weight=1.0):
    n = len(d)
    oof = dict(p_answered=np.full(n, np.nan),
               conf_probs=np.full((n, 5), np.nan),
               engagement=np.full(n, np.nan))
    sig_a, sig_b = [], []
    if split == "question":
        splitter, groups = GroupKFold(5), d.question_id.values
    elif split == "student":
        splitter, groups = GroupKFold(5), d.student_uid.values
    elif split == "attempt":
        splitter, groups = GroupKFold(5), d.attempt_id.values
    else:
        splitter, groups = KFold(5, shuffle=True, random_state=seed), None

    t0 = time.time()
    for k, (tr, te) in enumerate(splitter.split(d, groups=groups)):
        r = fit_predict(d, emb_cols, tr, te, epochs=epochs, seed=seed, verbose=verbose,
                        careless_weight=careless_weight)
        oof["p_answered"][te] = r["p_answered"]
        oof["conf_probs"][te] = r["conf_probs"]
        oof["engagement"][te] = r["engagement"]
        sig_a.append(r["sigma_a"]); sig_b.append(r["sigma_b"])
        print(f"    fold {k+1}/5 done ({time.time()-t0:.0f}s)  "
              f"sigma_student={r['sigma_a']:.3f} sigma_question={r['sigma_b']:.3f}")
    oof["sigma_a"], oof["sigma_b"] = float(np.mean(sig_a)), float(np.mean(sig_b))
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["student", "attempt", "question", "random"],
                    choices=["question", "student", "attempt", "random"])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--embedding", default="SentenceBERT",
                    choices=["SentenceBERT", "BERT", "RoBERTa"])
    ap.add_argument("--quick", action="store_true", help="200 epochs, question split only")
    ap.add_argument("--out", default="idea1_results")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tier", default="none", choices=["none","T1","T2","T3"],
                    help="restrict to an engagement tier before fitting")
    ap.add_argument("--careless-weight", type=float, default=0.15,
                    help="training weight on careless rows in the ordinal loss "
                         "(1.0 = off). Measured optimum ~0.1-0.3.")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.splits, a.seeds = 200, ["question"], [0]

    print("=" * 78)
    print("IDEA 1 — Hierarchical Ordinal Model + Careless Mixture")
    print("=" * 78)
    print("\n[1/4] Loading data ...")
    B, Q, E = load_pooled(a.embedding)
    print("\n[2/4] Building features ...")
    d, emb_cols = build_features(B, Q, E)
    if a.tier != "none":
        print(f"\n[2b] Applying engagement tier filter: {a.tier}")
        d = apply_tier_filter(d, a.tier)

    os.makedirs(a.out, exist_ok=True)
    all_res = {}

    SPLIT_DOC = {
        "student":  ("unseen students — student effect falls back to CGPA prior",
                     "HONEST. No student or attempt leakage. Primary metric."),
        "attempt":  ("unseen attempts — student may be known from OTHER sittings",
                     "HONEST. No within-attempt leakage."),
        "question": ("unseen questions — same students/attempts ARE in train",
                     "*** INFLATED: 100% of test attempts also appear in training, so the "
                     "student effect can memorise that sitting's straightlined value. "
                     "Read the cold-student/cold-attempt rows instead. ***"),
        "random":   ("random rows — everything seen",
                     "*** LEAKY REFERENCE ONLY. Included to size the leakage gap. ***"),
    }
    for split in a.splits:
        desc, warn = SPLIT_DOC[split]
        print(f"\n[3/4] Fitting — split = COLD-{split.upper()} ({desc})")
        print(f"      {warn}")
        per_seed = []
        for sd in a.seeds:
            print(f"  seed {sd}:")
            oof = run_split(d, emb_cols, split, a.epochs, sd, a.verbose, a.careless_weight)
            clean = (~d.eval_careless.values)
            for nm, mk in [("all", np.ones(len(d), bool)), ("clean", clean),
                           ("careless", ~clean)]:
                res, tables = evaluate(d, oof, nm, mk, f"{split}/{nm}")
                res["_split"], res["_subset"], res["_seed"] = split, nm, sd
                res["sigma_student"], res["sigma_question"] = oof["sigma_a"], oof["sigma_b"]
                per_seed.append(res)
                if nm == "clean":
                    for tn, tv in tables.items():
                        p = os.path.join(a.out, f"{split}_{nm}_seed{sd}_{tn}.csv")
                        (tv if isinstance(tv, pd.DataFrame) else pd.DataFrame(tv)).to_csv(p, index=False)
        all_res[split] = per_seed

    # ---------------- report ----------------
    print("\n[4/4] Results")
    rows = [r for v in all_res.values() for r in v]
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "all_metrics.csv"), index=False)

    key = ["_split", "_subset", "n_eval", "mae", "baseline_mean_mae", "lift_mae_vs_mean",
           "rmse", "spearman", "qwk", "adjacent_acc",
           "conf5_accuracy", "conf5_f1_macro", "conf5_bal_f1_macro",
           "conf3_accuracy", "conf3_f1_macro", "conf3_bal_f1_macro",
           "conf3_bal_accuracy", "conf3_balanced_acc",
           "hurdle_f1_macro", "hurdle_roc_auc", "engage_f1_macro", "engage_roc_auc",
           "ece", "sigma_student", "sigma_question"]
    key = [c for c in key if c in R.columns]
    agg = R.groupby(["_split", "_subset"])[
        [c for c in key if c not in ("_split", "_subset")]].mean().round(4)

    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\n--- HEADLINE (mean over seeds) ---")
    print("  Read the 'clean' subset rows: careless rows are trivially predictable")
    print("  (constant target) and inflate every pooled metric.")
    print("  Trust split order: student > attempt >> question > random.\n")
    print(agg.to_string())

    print("\n--- PER-CLASS breakdown, clean rows, first split/seed ---")
    print("  'conf5/conf3'      = argmax rule (collapses to majority rung on weak signal)")
    print("  'conf5_bal/conf3_bal' = prior-corrected rule -> the informative P/R/F1")
    for f in ["conf5_per_class", "conf5_bal_per_class",
              "conf3_per_class", "conf3_bal_per_class",
              "hurdle_per_class", "engage_per_class"]:
        p = os.path.join(a.out, f"{a.splits[0]}_clean_seed{a.seeds[0]}_{f}.csv")
        if os.path.exists(p):
            print(f"\n[{f}]")
            print(pd.read_csv(p).round(4).to_string(index=False))

    print(f"\n--- VARIANCE COMPONENTS (the 53% / 7% story) ---")
    for sp in all_res:
        r = [x for x in all_res[sp] if x["_subset"] == "clean"][0]
        sa, sb = r["sigma_student"], r["sigma_question"]
        print(f"  {sp:9s}  sigma_student={sa:.3f}  sigma_question={sb:.3f}  "
              f"ratio={sa/max(sb,1e-9):.2f}x  -> student effect dominates" if sa > sb else
              f"  {sp:9s}  sigma_student={sa:.3f}  sigma_question={sb:.3f}")

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({k: [{kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                        for kk, vv in r.items()} for r in v] for k, v in all_res.items()},
                  f, indent=2)
    print(f"\nSaved -> {a.out}/all_metrics.csv, summary.json, per-class CSVs")


if __name__ == "__main__":
    main()
