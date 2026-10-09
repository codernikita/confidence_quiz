/**
 * AWS Lambda entry point for the guidance API, served through a function URL.
 *
 * Same chain, prompt, link checks and error wording as the local server
 * (server/index.mjs); only the transport differs. Deployed by
 * scripts/deploy_guidance_lambda.sh, which bundles this file with esbuild.
 *
 * Configuration is Lambda environment variables:
 *   GEMINI_API_KEY          required
 *   FIREBASE_PROJECT_ID     when set, every call needs a valid Firebase ID token
 *   GEMINI_MODEL, GEMINI_FALLBACK_MODELS, GEMINI_THINKING_LEVEL   optional
 *
 * CORS is configured on the function URL itself (allowed origin = the site),
 * so this handler sets no CORS headers of its own; Lambda answers preflights.
 */
import {
  DEFAULT_MODEL,
  DEFAULT_FALLBACKS,
  DEFAULT_THINKING_LEVEL,
  GuidanceRequest,
  createGuidanceChains,
  describeFailure,
  generateGuidance,
} from "./guidance.mjs";
import { verifyFirebaseToken } from "./auth.mjs";

const MAX_BODY = 256 * 1024;
// Stop this long before Lambda's own timeout so the student gets a readable
// error rather than a bare 502 from a killed function.
const HEADROOM_MS = 8_000;

const env = process.env;
const API_KEY = String(env.GEMINI_API_KEY || "").trim();
const PROJECT_ID = String(env.FIREBASE_PROJECT_ID || "").trim();
const MODELS = [
  env.GEMINI_MODEL || DEFAULT_MODEL,
  ...(env.GEMINI_FALLBACK_MODELS !== undefined
    ? env.GEMINI_FALLBACK_MODELS.split(",").map((m) => m.trim()).filter(Boolean)
    : DEFAULT_FALLBACKS),
];
const THINKING = env.GEMINI_THINKING_LEVEL ?? DEFAULT_THINKING_LEVEL;

// Built once per container, reused across warm invocations.
const chains = API_KEY
  ? createGuidanceChains({ apiKey: API_KEY, models: MODELS, thinkingLevel: THINKING })
  : null;

const json = (statusCode, body) => ({
  statusCode,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(body),
});

export async function handler(event, context) {
  const method = event.requestContext?.http?.method || "GET";
  const path = (event.rawPath || "/").replace(/\/+$/, "") || "/";

  if (method === "GET" && (path === "/api/health" || path === "/")) {
    return json(200, { ok: true, models: MODELS, hasKey: Boolean(API_KEY), auth: Boolean(PROJECT_ID) });
  }
  if (method !== "POST" || (path !== "/api/guidance" && path !== "/")) {
    return json(404, { error: "Not found" });
  }
  if (!chains) {
    return json(503, { error: "The guidance service is not configured (no Gemini key)." });
  }

  if (PROJECT_ID) {
    try {
      await verifyFirebaseToken(event.headers?.authorization, PROJECT_ID);
    } catch (err) {
      return json(err.status || 401, { error: err.message });
    }
  }

  const raw = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64").toString("utf8")
    : event.body || "";
  if (raw.length > MAX_BODY) return json(413, { error: "Request body too large" });

  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "Body is not valid JSON" });
  }
  const parsed = GuidanceRequest.safeParse(body);
  if (!parsed.success) {
    return json(400, {
      error: "Request does not match the expected shape",
      issues: parsed.error.issues.slice(0, 5),
    });
  }

  const budget = Math.max(5_000, (context?.getRemainingTimeInMillis?.() ?? 180_000) - HEADROOM_MS);
  const signal = AbortSignal.timeout(budget);
  const started = Date.now();
  try {
    const { guidance, model, linkStats } = await generateGuidance(chains, parsed.data, {
      signal,
      onFallback: (m, err) =>
        console.warn(`${m} unavailable (${err?.status || err?.lc_error_code || "error"}), trying the next model`),
    });
    console.log(
      JSON.stringify({
        event: "guidance",
        questions: parsed.data.questions.length,
        model,
        seconds: Number(((Date.now() - started) / 1000).toFixed(1)),
        links: linkStats,
      }),
    );
    return json(200, { guidance, model, generatedAt: new Date().toISOString() });
  } catch (err) {
    console.error("guidance failed:", err?.message || err);
    return signal.aborted
      ? json(504, { error: "Gemini took too long to respond. Try again." })
      : json(502, { error: describeFailure(err, MODELS) });
  }
}
