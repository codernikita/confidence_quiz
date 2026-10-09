/**
 * Client for the study-guidance API (server/index.mjs).
 *
 * Sends each question together with the two labels already computed in the
 * browser -- the behavioural state and the predicted confidence -- and gets
 * back Gemini's per-question diagnosis plus an attempt-level summary.
 *
 * Nothing identifying leaves the browser here: no roll number, email, name or
 * CGPA. Gemini only needs the questions and how they were answered.
 *
 * VITE_GUIDANCE_API_URL points at a separately hosted API in production; the
 * default relies on the Vite dev/preview proxy.
 *
 * The student's Firebase ID token goes along as a bearer token. The deployed
 * API (server/lambda.mjs) rejects calls without one, so its public URL cannot
 * be used to spend the Gemini quota by anyone not signed in to this quiz.
 */
import { auth } from "../config/firebase";

const ENDPOINT = import.meta.env.VITE_GUIDANCE_API_URL || "/api/guidance";

// In development the Vite proxy serves /api. A production build has no such
// route (CloudFront would answer with the SPA's index.html), so without an
// explicit API URL the results page simply leaves guidance out.
export const GUIDANCE_ENABLED =
  Boolean(import.meta.env.VITE_GUIDANCE_API_URL) || Boolean(import.meta.env.DEV);

export function buildGuidanceRequest(analysis) {
  const { items, calibration } = analysis;
  const meanTime =
    items.reduce((a, i) => a + (i.timeSpent || 0), 0) / (items.length || 1);

  return {
    attempt: {
      score: items.filter((i) => i.isCorrect).length,
      total: items.length,
      meanTimeSec: Math.round(meanTime),
      calibrationReliability: calibration.reliability,
    },
    questions: items.map((it, i) => ({
      questionId: it.questionId,
      number: i + 1,
      block: it.isCalibration ? "calibration" : "technical",
      type: it.type || "",
      difficulty: it.difficulty || "medium",
      text: it.text,
      prompt: it.prompt || "",
      options: it.options || [],
      selectedOption: it.selectedOption || null,
      correctAnswer: it.correctAnswer,
      isCorrect: Boolean(it.isCorrect),
      explanation: it.explanation || "",
      behaviouralState: it.state,
      predictedConfidence: {
        label: it.prediction.label,
        value: Number(it.prediction.confidence.toFixed(2)),
        certainty: Number(it.prediction.certainty.toFixed(2)),
      },
      ratedConfidence: it.isCalibration ? it.actual : null,
      telemetry: {
        timeSpentSec: Math.round(it.timeSpent || 0),
        optionChanges: it.optionChanges || 0,
        markedForReview: Boolean(it.markedForReview),
        visits: it.visits || 0,
      },
    })),
  };
}

export async function fetchGuidance(analysis, { signal } = {}) {
  const headers = { "Content-Type": "application/json" };
  // getIdToken() refreshes an expired token itself; a failure here just means
  // no header, and the server explains that the student needs to sign in.
  const token = await auth?.currentUser?.getIdToken().catch(() => null);
  if (token) headers.Authorization = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(ENDPOINT, {
      method: "POST",
      headers,
      body: JSON.stringify(buildGuidanceRequest(analysis)),
      signal,
    });
  } catch (err) {
    if (err?.name === "AbortError") throw err;
    throw new Error("Could not reach the guidance server. Is it running?");
  }

  const body = await res.json().catch(() => null);
  if (!res.ok || !body?.guidance) {
    throw new Error(body?.error || `Guidance server returned ${res.status}`);
  }
  return body;
}
