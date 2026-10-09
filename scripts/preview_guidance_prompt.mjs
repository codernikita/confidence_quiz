/**
 * Shows exactly what Gemini is sent for study guidance, without taking the quiz.
 *
 *   npm run guidance:prompt              # the formatted prompt for a sample attempt
 *   npm run guidance:prompt -- --schema  # the JSON schema Gemini must answer in
 *   npm run guidance:prompt -- --live    # actually call Gemini (needs GEMINI_API_KEY)
 *
 * The sample attempt is the bundled question bank answered with a deliberate
 * mix of every behavioural state, scored by the real browser model
 * (src/model/predict.js) and turned into a request by the real client code
 * (src/lib/guidance.js) -- so this is the same payload the app sends.
 */
import { build } from "esbuild";
import { writeFileSync, mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { toJSONSchema } from "zod";
import {
  DEFAULT_MODEL,
  DEFAULT_FALLBACKS,
  DEFAULT_THINKING_LEVEL,
  GuidanceRequest,
  GuidanceResponse,
  guidancePrompt,
  promptInput,
  createGuidanceChains,
  generateGuidance,
} from "../server/guidance.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const args = new Set(process.argv.slice(2));

if (args.has("--schema")) {
  console.log(JSON.stringify(toJSONSchema(GuidanceResponse), null, 2));
  process.exit(0);
}

// The browser modules import JSON and read import.meta.env; bundle them for Node
// the same way scripts/parity_test.mjs does.
const dir = mkdtempSync(join(tmpdir(), "guidance-"));
const entry = join(dir, "entry.js");
writeFileSync(
  entry,
  `export { scoreAttempt } from ${JSON.stringify(join(ROOT, "src/model/predict.js"))};
   export { buildGuidanceRequest } from ${JSON.stringify(join(ROOT, "src/lib/guidance.js"))};
   export { CALIBRATION_QUESTIONS, TECHNICAL_QUESTIONS } from ${JSON.stringify(join(ROOT, "src/data/questions.js"))};`,
);
const out = join(dir, "bundle.mjs");
await build({
  entryPoints: [entry],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  loader: { ".json": "json" },
  define: { "import.meta.env": "{}" },
  // The client attaches a Firebase token when signed in; nobody is here.
  plugins: [
    {
      name: "no-firebase",
      setup(b) {
        b.onResolve({ filter: /config\/firebase$/ }, () => ({ path: "firebase-stub", namespace: "stub" }));
        b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export const auth = null;" }));
      },
    },
  ],
  logLevel: "silent",
});
const { scoreAttempt, buildGuidanceRequest, CALIBRATION_QUESTIONS, TECHNICAL_QUESTIONS } =
  await import(pathToFileURL(out).href);

/* One row per question: [picked option index or null, seconds, option changes,
 * flagged].  Chosen to hit every state: mastery, shaky, misconception (fast and
 * wrong), confusion (slow, changed, wrong) and one skipped. */
const PLAN = {
  cal_1: [0, 40, 0, false], // "Valid" -- wrong, fast: misconception
  cal_2: [1, 70, 1, false], // correct after a change: shaky
  cal_3: [0, 25, 0, false], // correct, fast: mastery
  tech_1: [2, 30, 0, false], // 3NF -- wrong, fast: misconception
  tech_2: [1, 20, 0, false], // correct: mastery
  tech_3: [0, 95, 2, true], // correct after changes: shaky
  tech_4: [0, 30, 0, false],
  tech_5: [1, 110, 2, true], // dirty read -- wrong, hesitated: confusion
  tech_6: [1, 25, 0, false], // "every query faster" -- misconception
  tech_7: [0, 18, 0, false],
  tech_8: [2, 80, 1, false], // 20,000 -- confusion
  tech_9: [0, 22, 0, false],
  tech_10: [null, 60, 0, true], // skipped
  tech_11: [2, 28, 0, false], // "distinct values only" -- misconception
  tech_12: [0, 75, 1, false],
};
const RATINGS = { cal_1: 5, cal_2: 3, cal_3: 4 };

const row = (q, block) => {
  const [pick, t, changes, flagged] = PLAN[q.id];
  const selected = pick === null ? "" : q.options[pick];
  return {
    questionId: q.id,
    block,
    type: q.type,
    difficulty: q.difficulty,
    text: q.text,
    prompt: q.prompt || "",
    options: q.options,
    correctAnswer: q.correctAnswer,
    explanation: q.explanation,
    selectedOption: selected,
    isCorrect: Boolean(selected) && selected === q.correctAnswer,
    timeSpent: t,
    optionChanges: changes,
    markedForReview: flagged,
    reviewClickCount: flagged ? 1 : 0,
    visits: flagged ? 2 : 1,
    confidenceRating: RATINGS[q.id] ?? 0,
    isCalibration: block === "calibration",
  };
};

const analysis = scoreAttempt({
  calibrationItems: CALIBRATION_QUESTIONS.map((q) => row(q, "calibration")),
  technicalItems: TECHNICAL_QUESTIONS.map((q) => row(q, "technical")),
  cgpa: null,
});

// Validate with the server's own schema, so a drift between client and server
// shows up here rather than as a 400 in the browser.
const request = GuidanceRequest.parse(buildGuidanceRequest(analysis));

if (!args.has("--live")) {
  const messages = await guidancePrompt.formatMessages(promptInput(request));
  for (const m of messages) {
    console.log(`\n=============== ${m.getType().toUpperCase()} ===============\n`);
    console.log(m.content);
  }
  console.error(
    `\n[${request.questions.length} questions, ~${Math.round(
      messages.reduce((a, m) => a + m.content.length, 0) / 4,
    )} input tokens]`,
  );
  process.exit(0);
}

for (const file of [".env.local", ".env"]) {
  const p = join(ROOT, file);
  if (existsSync(p)) process.loadEnvFile(p);
}
const apiKey = (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "").trim();
if (!apiKey) {
  console.error("GEMINI_API_KEY is not set. Add it to .env.local first.");
  process.exit(1);
}
const models = [
  process.env.GEMINI_MODEL || DEFAULT_MODEL,
  ...(process.env.GEMINI_FALLBACK_MODELS !== undefined
    ? process.env.GEMINI_FALLBACK_MODELS.split(",").map((m) => m.trim()).filter(Boolean)
    : DEFAULT_FALLBACKS),
];
console.error(`Calling ${models.join(" > ")} with ${request.questions.length} questions…`);
const started = Date.now();
const { guidance, model, linkStats } = await generateGuidance(
  createGuidanceChains({ apiKey, models, thinkingLevel: process.env.GEMINI_THINKING_LEVEL ?? DEFAULT_THINKING_LEVEL }),
  request,
  { onFallback: (m, err) => console.error(`${m} unavailable (${err?.status || "error"}), trying the next model`) },
);
console.log(JSON.stringify(guidance, null, 2));
console.error(`Done by ${model} in ${((Date.now() - started) / 1000).toFixed(1)}s; links ${JSON.stringify(linkStats)}`);
