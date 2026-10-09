/**
 * Guidance API -- the one piece of this app that needs a server, because it
 * holds the Gemini key.
 *
 *   POST /api/guidance   attempt in, study guidance out (see guidance.mjs)
 *   GET  /api/health     { ok, models, hasKey }
 *
 * In development Vite proxies /api here (vite.config.js), so the browser only
 * ever talks to its own origin.  Reads GEMINI_API_KEY from .env.local or .env;
 * neither file is committed and neither is exposed to the bundle, because Vite
 * only ships VITE_-prefixed variables.
 *
 * The env files are re-read on every request, so adding or changing the key
 * takes effect without a restart -- `node --watch` only restarts on code
 * changes, and a server that silently kept running keyless was the first thing
 * that broke when this was set up.
 *
 * Starts without a key on purpose: the quiz must keep working, so a missing key
 * becomes a 503 with a readable reason that the results page shows in place of
 * the guidance.
 */
import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { parseEnv } from "node:util";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  DEFAULT_MODEL,
  DEFAULT_FALLBACKS,
  DEFAULT_THINKING_LEVEL,
  GuidanceRequest,
  describeFailure,
  createGuidanceChains,
  generateGuidance,
} from "./guidance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHELL_ENV = { ...process.env };
const list = (v) =>
  String(v || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);

// .env.local overrides .env; anything exported in the shell overrides both.
function readEnv() {
  const fromFiles = {};
  for (const file of [".env", ".env.local"]) {
    const p = path.join(ROOT, file);
    if (existsSync(p)) Object.assign(fromFiles, parseEnv(readFileSync(p, "utf8")));
  }
  return { ...fromFiles, ...SHELL_ENV };
}

function currentConfig() {
  const env = readEnv();
  return {
    apiKey: String(env.GEMINI_API_KEY || env.GOOGLE_API_KEY || "").trim(),
    models: [
      env.GEMINI_MODEL || DEFAULT_MODEL,
      ...(env.GEMINI_FALLBACK_MODELS !== undefined
        ? list(env.GEMINI_FALLBACK_MODELS)
        : DEFAULT_FALLBACKS),
    ],
    // Empty string = leave thinking at the model's own default.
    thinkingLevel: env.GEMINI_THINKING_LEVEL ?? DEFAULT_THINKING_LEVEL,
  };
}

// Rebuilt only when the key or the model list actually changes.
let cache = { signature: null, chains: null };
function chainsFor({ apiKey, models, thinkingLevel }) {
  const signature = `${apiKey}|${models.join(",")}|${thinkingLevel}`;
  if (cache.signature !== signature) {
    cache = {
      signature,
      chains: apiKey ? createGuidanceChains({ apiKey, models, thinkingLevel }) : null,
    };
  }
  return cache.chains;
}

const PORT = Number(process.env.GUIDANCE_PORT || 8787);
const MAX_BODY = 256 * 1024;
// Long enough for a fallback after an overloaded primary plus a full answer.
const TIMEOUT_MS = 180_000;
// Only needed when the API is served from a different origin than the site.
const ALLOWED_ORIGINS = list(process.env.GUIDANCE_ALLOWED_ORIGINS);

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(Object.assign(new Error("Request body too large"), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch {
        reject(Object.assign(new Error("Body is not valid JSON"), { status: 400 }));
      }
    });
    req.on("error", reject);
  });
}

async function handleGuidance(req, res) {
  const config = currentConfig();
  const chains = chainsFor(config);
  if (!chains) {
    return send(res, 503, {
      error: "GEMINI_API_KEY is not set on the guidance server. Add it to .env.local.",
    });
  }

  let body;
  try {
    body = await readJson(req);
  } catch (err) {
    return send(res, err.status || 400, { error: err.message });
  }

  const parsed = GuidanceRequest.safeParse(body);
  if (!parsed.success) {
    return send(res, 400, {
      error: "Request does not match the expected shape",
      issues: parsed.error.issues.slice(0, 5),
    });
  }

  // Stop paying for a generation nobody is waiting for. `res` close (not `req`)
  // is the reliable client-disconnect signal once the body has been read.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("timeout")), TIMEOUT_MS);
  res.on("close", () => {
    if (!res.writableEnded) controller.abort(new Error("client disconnected"));
  });

  const started = Date.now();
  try {
    const { guidance, model, linkStats } = await generateGuidance(chains, parsed.data, {
      signal: controller.signal,
      onFallback: (m, err) =>
        console.warn(
          `[guidance] ${m} unavailable (${err?.status || err?.lc_error_code || "error"}), trying the next model`,
        ),
    });
    console.log(
      `[guidance] ${parsed.data.questions.length} questions by ${model} in ${((Date.now() - started) / 1000).toFixed(1)}s; ` +
        `links: ${linkStats.verified}/${linkStats.total} verified, ` +
        `${linkStats.youtubeSearch} YouTube searches, ${linkStats.webSearch} web searches`,
    );
    send(res, 200, { guidance, model, generatedAt: new Date().toISOString() });
  } catch (err) {
    if (controller.signal.aborted && res.destroyed) return;
    console.error("[guidance] failed:", err?.message || err);
    const timedOut = controller.signal.aborted;
    send(res, timedOut ? 504 : 502, {
      error: timedOut
        ? "Gemini took too long to respond. Try again."
        : describeFailure(err, config.models),
    });
  } finally {
    clearTimeout(timer);
  }
}

const server = createServer(async (req, res) => {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  }
  if (req.method === "OPTIONS") return res.writeHead(204).end();

  const url = new URL(req.url, "http://localhost");
  if (req.method === "GET" && url.pathname === "/api/health") {
    const { apiKey, models, thinkingLevel } = currentConfig();
    return send(res, 200, { ok: true, models, thinkingLevel, hasKey: Boolean(apiKey) });
  }
  if (req.method === "POST" && url.pathname === "/api/guidance") {
    return handleGuidance(req, res);
  }
  send(res, 404, { error: "Not found" });
});

server.listen(PORT, () => {
  const { apiKey, models } = currentConfig();
  console.log(`[guidance] listening on http://localhost:${PORT}  models=${models.join(" > ")}`);
  if (!apiKey) {
    console.warn(
      "[guidance] GEMINI_API_KEY is not set yet -- the quiz will run without study guidance. " +
        "Add it to .env.local; no restart needed.",
    );
  }
});
