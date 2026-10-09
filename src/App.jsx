import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import SignIn from "./components/SignIn";
import Landing from "./components/Landing";
import CalibrationStage from "./components/CalibrationStage";
import TechnicalStage from "./components/TechnicalStage";
import Results from "./components/Results";
import { createTelemetry } from "./lib/telemetry";
import { loadBank, saveAttempt, countPriorAttempts } from "./lib/firestore";
import { fetchGuidance, GUIDANCE_ENABLED } from "./lib/guidance";
import {
  watchAuth,
  resolveRedirect,
  signOutUser,
  isConfigured,
} from "./config/firebase";
import { scoreAttempt, MODEL_META } from "./model/predict";
import { DEFAULT_CONFIG } from "./data/questions";

const STAGES = {
  LOADING: "loading",
  SIGNIN: "signin",
  LANDING: "landing",
  CALIBRATION: "calibration",
  TECHNICAL: "technical",
  SCORING: "scoring",
  RESULTS: "results",
};

export default function App() {
  const [stage, setStage] = useState(STAGES.LOADING);
  const [bank, setBank] = useState(null);
  const [account, setAccount] = useState(null); // { uid, email, name, photo }
  const [authError, setAuthError] = useState(null);
  const [priorAttempts, setPriorAttempts] = useState(0);
  const [student, setStudent] = useState({ rollNumber: "", cgpa: null });
  const [analysis, setAnalysis] = useState(null);
  const [saveState, setSaveState] = useState("idle");
  // { status: "idle" | "loading" | "ready" | "error", data, error }
  const [guidance, setGuidance] = useState({ status: "idle" });
  const [timeLeft, setTimeLeft] = useState(null);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("bisq-theme") || "system",
  );

  const telemetry = useRef(createTelemetry()).current;
  const startedAt = useRef(null);
  // A quiz in progress must survive an auth token refresh without being reset.
  const stageRef = useRef(stage);
  stageRef.current = stage;

  /* ---------------------------------------------------- boot */
  useEffect(() => {
    let alive = true;

    loadBank().then((loaded) => {
      if (!alive) return;
      setBank(loaded);
      // Without Firebase there is nobody to sign in, so go straight to the
      // quiz. This has to happen here rather than in a separate effect keyed
      // on `stage`, or the auth listener below races it to SIGNIN first.
      if (!isConfigured) setStage(STAGES.LANDING);
    });

    if (!isConfigured) {
      return () => {
        alive = false;
      };
    }

    // Collect the result of a redirect sign-in before wiring up the listener,
    // so a redirect return does not flash the sign-in screen.
    resolveRedirect().catch((err) => alive && setAuthError(err.message));

    const stop = watchAuth((profile, err) => {
      if (!alive) return;
      if (err) setAuthError(err.message);

      setAccount(profile);
      if (profile) {
        setAuthError(null);
        countPriorAttempts(profile.uid).then((n) => alive && setPriorAttempts(n));

        // Re-fetch the bank now that we are authenticated. The boot-time call
        // above runs before sign-in, so the `request.auth != null` rule denies
        // it and it falls back to the bundled bank -- meaning the seeded
        // Firestore questions would otherwise never be used at all.
        loadBank().then((fresh) => {
          if (!alive || fresh.source !== "firestore") return;
          // Never swap the bank out from under a quiz in progress.
          if (
            stageRef.current === STAGES.LOADING ||
            stageRef.current === STAGES.SIGNIN ||
            stageRef.current === STAGES.LANDING
          ) {
            setBank(fresh);
          }
        });
        // Only advance if we are still on a pre-quiz screen. A token refresh
        // mid-quiz re-fires this listener and must not restart the attempt.
        if (
          stageRef.current === STAGES.LOADING ||
          stageRef.current === STAGES.SIGNIN
        ) {
          setStage(STAGES.LANDING);
        }
      } else if (stageRef.current !== STAGES.RESULTS) {
        setStage(STAGES.SIGNIN);
      }
    });

    return () => {
      alive = false;
      stop();
    };
  }, []);

  /* ---------------------------------------------------- theme */
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    localStorage.setItem("bisq-theme", theme);
  }, [theme]);

  /* ---------------------------------------------------- focus tracking
   * Time only accrues while the tab is visible and focused. Without this a
   * student who alt-tabs away looks like deep deliberation, and time is one of
   * the model's inputs. */
  useEffect(() => {
    const onVis = () =>
      document.visibilityState === "hidden" ? telemetry.pause() : telemetry.resume();
    const onBlur = () => telemetry.pause();
    const onFocus = () => telemetry.resume();

    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
  }, [telemetry]);

  /* ---------------------------------------------------- accidental exit */
  useEffect(() => {
    const inProgress =
      stage === STAGES.CALIBRATION || stage === STAGES.TECHNICAL;
    if (!inProgress) return;
    const warn = (e) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [stage]);

  const config = bank?.config || DEFAULT_CONFIG;

  /* Gemini study guidance. Started from the submit handler rather than an
   * effect so StrictMode's double-mount does not send every attempt twice.
   * It never blocks the results page: the scores render at once and the
   * guidance fills in when it arrives. */
  const requestGuidance = useCallback((result) => {
    if (!GUIDANCE_ENABLED) return;
    setGuidance({ status: "loading" });
    fetchGuidance(result)
      .then((data) => setGuidance({ status: "ready", data }))
      .catch((err) => setGuidance({ status: "error", error: err.message }));
  }, []);

  const submit = useCallback(async () => {
    setStage(STAGES.SCORING);
    telemetry.pause();

    const calibrationItems = telemetry
      .finalise(bank.calibration)
      .map((r) => ({ ...r, isCalibration: true }));
    const technicalItems = telemetry
      .finalise(bank.technical)
      .map((r) => ({ ...r, isCalibration: false }));

    const result = scoreAttempt({
      calibrationItems,
      technicalItems,
      cgpa: student.cgpa,
    });
    setAnalysis(result);
    requestGuidance(result);

    const payload = {
      rollNumber: student.rollNumber,
      studentCgpa: student.cgpa,
      studentUid: account?.uid || null,
      studentEmail: account?.email || "",
      studentName: account?.name || "",
      // Retakes are allowed but flagged: filter to attemptNumber == 1 to keep
      // only first sittings, which is what the training data needed.
      attemptNumber: priorAttempts + 1,
      isRetake: priorAttempts > 0,
      timestamp: new Date().toISOString(),
      startedAt: startedAt.current,
      questionSource: bank.source,
      modelVersion: MODEL_META.generated_by,
      calibrationRatings: result.calibration.ratings,
      calibrationLooMae: Number(result.calibration.looMae.toFixed(4)),
      calibrationReliability: result.calibration.reliability,
      correctCount: result.items.filter((i) => i.isCorrect).length,
      questionCount: result.items.length,
      correctPercentage:
        Math.round(
          (result.items.filter((i) => i.isCorrect).length / result.items.length) *
            10000,
        ) / 100,
      // one entry per question -- same field names as the existing Bis-quiz
      // export so the analysis scripts keep working unchanged
      behavioralMetrics: Object.fromEntries(
        result.items.map((i) => [
          i.questionId,
          {
            block: i.block,
            finalSelectedOption: i.selectedOption,
            timeSpent: i.timeSpent,
            optionChanges: i.optionChanges,
            markedForReview: i.markedForReview,
            reviewClickCount: i.reviewClickCount,
            visits: i.visits,
            confidenceRating: i.isCalibration ? i.actual : 0,
            isCorrect: i.isCorrect,
            predictedConfidence: Number(i.prediction.confidence.toFixed(4)),
            predictedClass: i.prediction.label,
            predictedProbs: i.prediction.probs.map((p) => Number(p.toFixed(4))),
            predictedSd: Number(i.prediction.sd.toFixed(4)),
            behaviouralState: i.state,
          },
        ]),
      ),
    };

    if (isConfigured && account?.uid) {
      const res = await saveAttempt(payload);
      setSaveState(res.ok ? "saved" : "failed");
    } else {
      setSaveState("failed");
    }

    setStage(STAGES.RESULTS);
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [bank, student, telemetry, account, priorAttempts, requestGuidance]);

  /* ---------------------------------------------------- timer
   * Gates the whole sitting; expiry force-submits whatever is there. */
  useEffect(() => {
    if (stage !== STAGES.CALIBRATION && stage !== STAGES.TECHNICAL) return;
    if (timeLeft === null) return;
    if (timeLeft <= 0) {
      submit();
      return;
    }
    const t = setTimeout(() => setTimeLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [stage, timeLeft, submit]);

  const counts = useMemo(
    () => ({
      calibration: bank?.calibration.length || 0,
      technical: bank?.technical.length || 0,
      total: (bank?.calibration.length || 0) + (bank?.technical.length || 0),
    }),
    [bank],
  );

  if (!bank || stage === STAGES.LOADING) {
    return (
      <div className="center-screen">
        <div style={{ textAlign: "center" }}>
          <div className="spinner" />
          <p className="muted">Loading…</p>
        </div>
      </div>
    );
  }

  if (stage === STAGES.SIGNIN) {
    return (
      <div className="app">
        <TopBar config={config} stage={stage} theme={theme} setTheme={setTheme} />
        <SignIn config={config} counts={counts} authError={authError} />
      </div>
    );
  }

  if (stage === STAGES.SCORING) {
    return (
      <div className="center-screen">
        <div style={{ textAlign: "center" }}>
          <div className="spinner" />
          <p className="muted">Scoring your attempt…</p>
        </div>
      </div>
    );
  }

  const inQuiz = stage === STAGES.CALIBRATION || stage === STAGES.TECHNICAL;

  return (
    <div className="app">
      <TopBar
        config={config}
        stage={stage}
        theme={theme}
        setTheme={setTheme}
        timeLeft={inQuiz ? timeLeft : null}
        account={stage === STAGES.LANDING ? account : null}
        onSignOut={stage === STAGES.LANDING ? signOutUser : null}
      />

      {stage === STAGES.LANDING && (
        <Landing
          config={config}
          counts={counts}
          offline={!isConfigured}
          account={account}
          priorAttempts={priorAttempts}
          onStart={(s) => {
            setStudent(s);
            startedAt.current = new Date().toISOString();
            setTimeLeft((config.totalTimeAllowedMinutes || 25) * 60);
            setStage(STAGES.CALIBRATION);
          }}
        />
      )}

      {stage === STAGES.CALIBRATION && (
        <CalibrationStage
          questions={bank.calibration}
          telemetry={telemetry}
          onComplete={() => {
            setStage(STAGES.TECHNICAL);
            window.scrollTo({ top: 0, behavior: "auto" });
          }}
        />
      )}

      {stage === STAGES.TECHNICAL && (
        <TechnicalStage
          questions={bank.technical}
          telemetry={telemetry}
          onSubmit={submit}
        />
      )}

      {stage === STAGES.RESULTS && analysis && (
        <Results
          analysis={analysis}
          saveState={saveState}
          rollNumber={student.rollNumber}
          attemptNumber={priorAttempts + 1}
          guidance={guidance}
          onRetryGuidance={() => requestGuidance(analysis)}
        />
      )}
    </div>
  );
}

function TopBar({ config, stage, theme, setTheme, timeLeft, account, onSignOut }) {
  const subtitle =
    stage === STAGES.CALIBRATION
      ? "Section 1 — confidence collected"
      : stage === STAGES.TECHNICAL
        ? "Section 2 — confidence predicted"
        : stage === STAGES.RESULTS
          ? "Your analysis"
          : config.subtitle;

  return (
    <header className="topbar">
      <div className="topbar__inner">
        <div className="topbar__title">
          {config.title}
          <span>{subtitle}</span>
        </div>

        {timeLeft !== null && timeLeft !== undefined && (
          <span className={`clock ${timeLeft < 120 ? "clock--urgent" : ""}`}>
            {String(Math.floor(timeLeft / 60)).padStart(2, "0")}:
            {String(timeLeft % 60).padStart(2, "0")}
          </span>
        )}

        {account && (
          <span className="whoami" title={account.email}>
            {account.photo ? (
              <img className="whoami__avatar" src={account.photo} alt="" />
            ) : (
              <span className="whoami__avatar whoami__avatar--fallback">
                {(account.name || account.email || "?")[0].toUpperCase()}
              </span>
            )}
            <span className="whoami__email">{account.email}</span>
            {onSignOut && (
              <button type="button" className="whoami__out" onClick={onSignOut}>
                Sign out
              </button>
            )}
          </span>
        )}

        <div className="seg" role="group" aria-label="Colour theme">
          {["light", "system", "dark"].map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={theme === t}
              onClick={() => setTheme(t)}
            >
              {t === "system" ? "Auto" : t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
