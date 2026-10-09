/**
 * Guidance API for free Node hosts (Render, etc.): a plain Node HTTP server.
 *
 * Same chain, prompt, link checks, Firebase-token check and error wording as
 * server/lambda.mjs; only the transport differs. Unlike server/index.mjs (local
 * development only) this one REQUIRES a valid Firebase ID token and sets CORS.
 *
 * Environment variables (set in the host's dashboard):
 *   GEMINI_API_KEY        required
 *   FIREBASE_PROJECT_ID   required; every call needs a token from this project
 *   ALLOWED_ORIGINS       optional, comma-separated extra origins (custom domain).
 *                         https://<project>.web.app, https://<project>.firebaseapp.com
 *                         and http://localhost:5173 are always allowed.
 *   PORT                  provided by the host
 *   GEMINI_MODEL, GEMINI_FALLBACK_MODELS, GEMINI_THINKING_LEVEL   optional
 */
import { createServer } from "node:http";
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

const env = process.env;
const list = (v) =>
  String(v || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

const PORT = Number(env.PORT || 10000);
const API_KEY = String(env.GEMINI_API_KEY || env.GOOGLE_API_KEY || "").trim();
const PROJECT_ID = String(env.FIREBASE_PROJECT_ID || "").trim();
const MODELS = [
  env.GEMINI_MODEL || DEFAULT_MODEL,
  ...(env.GEMINI_FALLBACK_MODELS !== undefined
    ? list(env.GEMINI_FALLBACK_MODELS)
    : DEFAULT_FALLBACKS),
];
const THINKING = env.GEMINI_THINKING_LEVEL ?? DEFAULT_THINKING_LEVEL;
const ALLOWED_ORIGINS = new Set([
  ...(PROJECT_ID
    ? [`https://${PROJECT_ID}.web.app`, `https://${PROJECT_ID}.firebaseapp.com`]
    : []),
  "http://localhost:5173",
  ...list(env.ALLOWED_ORIGINS),
]);

const MAX_BODY = 256 * 1024;
const TIMEOUT_MS = 170_000;

const chains = API_KEY
  ? createGuidanceChains({
      apiKey: API_KEY,
      models: MODELS,
      thinkingLevel: THINKING,
    })
  : null;

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(
          Object.assign(new Error("Request body too large"), { status: 413 }),
        );
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handleGuidance(req, res) {
  if (!chains) {
    return send(res, 503, {
      error: "The guidance service is not configured (no Gemini key).",
    });
  }
  // Fail closed: never run unauthenticated.
  if (!PROJECT_ID) {
    return send(res, 500, {
      error: "The guidance service is misconfigured (no project ID).",
    });
  }
  try {
    await verifyFirebaseToken(req.headers.authorization, PROJECT_ID);
  } catch (err) {
    return send(res, err.status || 401, { error: err.message });
  }

  let body;
  try {
    body = JSON.parse(await readBody(req));
  } catch (err) {
    return send(res, err.status || 400, {
      error: err.status ? err.message : "Body is not valid JSON",
    });
  }
  const parsed = GuidanceRequest.safeParse(body);
  if (!parsed.success) {
    return send(res, 400, {
      error: "Request does not match the expected shape",
      issues: parsed.error.issues.slice(0, 5),
    });
  }

  // Stop paying for a generation nobody is waiting for.
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error("timeout")),
    TIMEOUT_MS,
  );
  res.on("close", () => {
    if (!res.writableEnded) controller.abort(new Error("client disconnected"));
  });

  const started = Date.now();
  try {
    const { guidance, model, linkStats } = await generateGuidance(
      chains,
      parsed.data,
      {
        signal: controller.signal,
        onFallback: (m, err) =>
          console.warn(
            `${m} unavailable (${err?.status || err?.lc_error_code || "error"}), trying the next model`,
          ),
      },
    );
    console.log(
      JSON.stringify({
        event: "guidance",
        questions: parsed.data.questions.length,
        model,
        seconds: Number(((Date.now() - started) / 1000).toFixed(1)),
        links: linkStats,
      }),
    );
    send(res, 200, { guidance, model, generatedAt: new Date().toISOString() });
  } catch (err) {
    if (controller.signal.aborted && res.destroyed) return;
    console.error("guidance failed:", err?.message || err);
    send(res, controller.signal.aborted ? 504 : 502, {
      error: controller.signal.aborted
        ? "Gemini took too long to respond. Try again."
        : describeFailure(err, MODELS),
    });
  } finally {
    clearTimeout(timer);
  }
}

const server = createServer(async (req, res) => {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Authorization, Content-Type",
    );
    res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    res.setHeader("Access-Control-Max-Age", "86400");
  }
  if (req.method === "OPTIONS") return res.writeHead(204).end();

  const path =
    new URL(req.url, "http://localhost").pathname
      .replace(/\/{2,}/g, "/")
      .replace(/\/+$/, "") || "/";
  try {
    if (req.method === "GET" && (path === "/api/health" || path === "/")) {
      return send(res, 200, {
        ok: true,
        models: MODELS,
        hasKey: Boolean(API_KEY),
        auth: Boolean(PROJECT_ID),
      });
    }
    if (
      req.method === "POST" &&
      (path === "/api/guidance" || path === "/guidance" || path === "/")
    ) {
      return await handleGuidance(req, res);
    }
    send(res, 404, { error: "Not found" });
  } catch (err) {
    console.error("unhandled:", err?.message || err);
    if (!res.headersSent) send(res, 500, { error: "Internal error" });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(
    `guidance API listening on :${PORT}  models=${MODELS.join(" > ")}  auth=${Boolean(PROJECT_ID)}`,
  );
});
