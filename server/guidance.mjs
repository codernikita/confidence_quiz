/**
 * guidance.mjs -- LangChain + Gemini study guidance.
 *
 * One consolidated call per attempt: every question goes to Gemini together
 * with the two labels the app already computed for it, and Gemini returns a
 * per-question diagnosis (severity, topics, sources, next steps) plus an
 * attempt-level summary of which concepts are weak.  One call rather than one
 * per question so the topic names line up across questions and the overall
 * picture can be assembled from them.
 *
 * Runs server-side only.  The Gemini key must never reach the browser bundle --
 * the deployed site is public, and a key in the JS is a key anyone can use.
 */
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { checkResourceLinks } from "./resources.mjs";

export const DEFAULT_MODEL = "gemini-3.8-flash";
// Tried in order when the primary is overloaded or not available to this key.
// Availability swings minute to minute -- at one point all three Flash models
// answered 503 "high demand" together -- so the list is long, and ends on a
// Flash-Lite model, which is less capable but rarely overloaded.
export const DEFAULT_FALLBACKS = [
  "gemini-3.5-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash-lite",
];
// Measured on a 15-question attempt with gemini-3.5-flash: "low" took 37s
// against 52s at the model default, with identical severities and equally
// specific diagnoses. Students wait on this, so speed wins.
export const DEFAULT_THINKING_LEVEL = "low";

/* ------------------------------------------------------------------ *
 * Request: what the browser sends.  Validated here rather than trusted,
 * and capped so a malformed client cannot run up a huge prompt.
 * ------------------------------------------------------------------ */
const str = (max) => z.string().max(max);

const QuestionIn = z.object({
  questionId: str(80),
  number: z.number().int().min(1).max(200),
  block: z.enum(["calibration", "technical"]),
  type: str(40).default(""),
  difficulty: str(20).default("medium"),
  text: str(4000),
  prompt: str(500).default(""),
  options: z.array(str(600)).max(10),
  selectedOption: str(600).nullable(),
  correctAnswer: str(600),
  isCorrect: z.boolean(),
  explanation: str(2000).default(""),
  behaviouralState: z.enum([
    "mastery",
    "shaky",
    "misconception",
    "confusion",
    "skipped",
  ]),
  predictedConfidence: z.object({
    label: z.enum(["Low", "Medium", "High"]),
    value: z.number().min(1).max(5),
    certainty: z.number().min(0).max(1),
  }),
  ratedConfidence: z.number().int().min(1).max(5).nullable(),
  telemetry: z.object({
    timeSpentSec: z.number().min(0),
    optionChanges: z.number().min(0),
    markedForReview: z.boolean(),
    visits: z.number().min(0),
  }),
});

export const GuidanceRequest = z.object({
  attempt: z.object({
    score: z.number().int().min(0),
    total: z.number().int().min(1),
    meanTimeSec: z.number().min(0),
    calibrationReliability: z.enum(["strong", "moderate", "weak", "flat"]),
  }),
  questions: z.array(QuestionIn).min(1).max(60),
});

/* ------------------------------------------------------------------ *
 * Response: what Gemini must return.  Passed to Gemini as a JSON schema
 * (responseSchema), so the shape is enforced at generation time and parsed
 * back through zod.  The .describe() strings travel with the schema and act
 * as field-level instructions.
 * ------------------------------------------------------------------ */
const Severity = z
  .enum(["none", "low", "moderate", "high"])
  .describe("Severity of the gap, per the rubric in the instructions.");

export const RESOURCE_KINDS = ["video", "article", "docs", "course", "practice", "book"];

const Resource = z.object({
  kind: z
    .enum(RESOURCE_KINDS)
    .describe(
      "video = a YouTube video; article = a blog post or tutorial page; docs = official documentation; course = free lecture series; practice = free problems or exercises; book = free online textbook chapter.",
    ),
  title: z
    .string()
    .describe("Exact title as published. For a video, the exact YouTube video title."),
  creator: z
    .string()
    .describe("YouTube channel, website, or author, exactly as named."),
  url: z
    .string()
    .describe(
      "Full URL of the specific page or video, as best you know it. Checked by the app before use.",
    ),
  why: z
    .string()
    .describe("One sentence: which part to watch or read, and what gap it fixes."),
});

const Topic = z.object({
  name: z.string().describe("Specific concept name, used verbatim everywhere else."),
  resources: z.array(Resource),
});

const QuestionGuidance = z.object({
  questionId: z.string().describe("Copy questionId exactly from the input."),
  severity: Severity,
  severityReason: z
    .string()
    .describe("One sentence citing the signals that set the severity."),
  diagnosis: z
    .string()
    .describe(
      "Two to three sentences, addressed to the student as 'you', on what their answer reveals about their understanding.",
    ),
  topics: z
    .array(z.string())
    .describe("Names of the topics this question tests, copied exactly from the topics list."),
  nextSteps: z
    .array(z.string())
    .describe("Concrete actions, each one sentence, most important first."),
});

export const GuidanceResponse = z.object({
  overall: z.object({
    summary: z
      .string()
      .describe("Three to four sentences addressed to the student: the overall pattern and where to focus."),
    weakConcepts: z
      .array(
        z.object({
          concept: z.string().describe("A topic name, copied exactly from the topics list."),
          severity: Severity,
          questionIds: z.array(z.string()),
          why: z.string().describe("One sentence on what went wrong with this concept."),
        }),
      )
      .describe("Concepts with a gap, most severe first. Empty if there are none."),
    strongConcepts: z
      .array(z.string())
      .describe("Topic names answered with mastery. Empty if there are none."),
    studyPlan: z
      .array(z.string())
      .describe("Three to six ordered steps, each one sentence, starting with the most severe gap."),
  }),
  topics: z
    .array(Topic)
    .describe("Every distinct concept the attempt touches, once each, with free resources."),
  questions: z.array(QuestionGuidance),
});

/* ------------------------------------------------------------------ *
 * The prompt.
 *
 * The system message explains where each label came from and how much to
 * trust it, because the two labels are very different things: the state is a
 * deterministic rule over what the student did, the confidence is a model
 * estimate that is right about two times in three.  Without saying so Gemini
 * would treat both as fact and tell students they "were confident" when the
 * model only guessed it.
 *
 * Literal braces are not allowed in a LangChain template (they are variable
 * slots), so the text below avoids them.  The attempt itself goes in as a
 * variable and is substituted verbatim.
 * ------------------------------------------------------------------ */
const SYSTEM = `You are a patient, precise study coach for university students. A student has just finished a quiz. For every question, you will diagnose what their answer reveals, judge how serious the gap is, and name the concepts involved. For every concept, you will point them to specific, free material to study. Then you will consolidate this into an overall picture.

# What you receive

The attempt as JSON. Every question has its full text, options, the student's answer, the correct answer, an explanation, and two labels the app has already computed.

1. behaviouralState -- a deterministic rule over what the student did. It is a fact, not an estimate.
   - mastery: correct, no hesitation
   - shaky: correct, but hesitated (changed option, flagged it, revisited, or took longer than their own average)
   - misconception: wrong, with no hesitation -- they likely hold a wrong belief and do not know it
   - confusion: wrong after visible hesitation -- they knew they were unsure
   - skipped: no answer

2. predictedConfidence -- a machine-learning estimate of how confident the student felt, as a class (Low, Medium, High), a value from 1 to 5, and the model's certainty in that class (0 to 1). The model is right about two times in three, so treat it as a soft signal: lean on it when certainty is high, and never present it to the student as fact. Say "you were likely confident", not "you were confident". For calibration questions ratedConfidence holds the rating the student actually gave; when present, trust it over the prediction.

attempt.calibrationReliability says how coherent the student's own ratings were: strong, moderate, weak, or flat (all ratings identical). If it is weak or flat, give predictedConfidence even less weight.

Calibration questions are general reasoning (logic) questions; technical questions are the subject being tested. Both get guidance.

# Severity rubric

Judge each question on these signals together, then pick one level:

- none: mastery. Correct and settled. Offer one way to go deeper, nothing remedial.
- low: shaky, or correct with Low predicted confidence. The answer was right but the understanding is fragile, and may have been a guess.
- moderate: confusion or skipped. A real gap the student is already aware of.
- high: misconception, especially with Medium or High confidence. A confidently held wrong belief is the most serious case, because the student will not go looking for the correction on their own.

Then adjust by one step where clearly warranted:
- Up one step if the question is easy or medium difficulty and the student got it wrong: a foundation is missing.
- Down one step if the question is hard and the wrong answer is a near-miss, i.e. a plausible distractor that shows partial understanding.
- The specific wrong option chosen is the strongest evidence of WHAT the student believes. Read it, and name the faulty belief in the diagnosis.

Never assign none to a wrong or skipped answer.

# What to write for each question

- diagnosis: address the student as "you". Name the specific idea they have right or wrong, using the option they picked. Do not repeat the question text back to them. Do not just restate the explanation; interpret it for this student.
- topics: the names of the two to four topics the question tests, copied exactly from the topics list below.
- nextSteps: one to three concrete actions, e.g. "Redo this question after writing out the candidate key first" rather than "Revise normalisation". Where it helps, point to one of the topic's resources by name.

# Topics

List every concept the attempt touches in topics, ONCE each -- one entry per distinct concept across the whole attempt, not one per question. Questions then refer to topics by name, so the same concept appearing in three questions is one topic with one set of resources. Use specific names ("Second normal form (partial dependency)", not "Databases"). Every weakConcepts.concept and strongConcepts entry must be one of these names, copied exactly.

# Resources for each topic

The student wants concrete, FREE material they can open right now: YouTube videos, blog posts and tutorials, documentation pages, free lecture series, free online textbook chapters, and free practice problems. Never suggest paid courses, paywalled or member-only articles, or books that must be bought.

How many, set by the worst severity among the questions that touch the topic:
- high or moderate: three or four -- at least one YouTube video, at least one written explanation (blog post, tutorial or documentation page), and a free practice resource when one exists.
- low: two -- one YouTube video and one written explanation.
- none: one, for going deeper.

Make each resource specific to its topic: a video on second normal form, not a full database course; a post explaining which anomalies READ COMMITTED allows, not a database's home page. Prefer creators with a strong reputation for the subject: established educational YouTube channels and university lecture recordings, well-known tutorial sites and engineering blogs, and official documentation. Within a topic, do not list two resources from the same creator.

For every resource:
- title: the exact title as published. For a video, the exact video title as it appears on YouTube.
- creator: the YouTube channel, website or author, exactly as named.
- url: the full address of that specific page or video, as best you know it -- the article itself, not the site's home page; for YouTube, the full watch URL. Always fill it in. The app opens every link before the student sees it; one that does not load, or loads a page about something else, is replaced by a search for the title and creator. So a wrong URL costs nothing, while an accurate title and creator always matter.
- why: one sentence saying which part to watch or read and what it will fix for this student.

# Overall section

- summary: three or four sentences to the student on the pattern across the quiz -- what they are solid on, where the serious gaps are, and anything notable about how their confidence lines up with correctness.
- weakConcepts: topics with a gap, most severe first, with the questions that showed it. A topic's severity is the worst severity among its questions.
- strongConcepts: topics answered with mastery.
- studyPlan: three to six ordered steps, most severe gap first, each a concrete action.

# Output rules

Return exactly one entry in questions for every input question, in input order, with questionId copied exactly. Be warm and direct, never condescending. Plain text only, no markdown inside the strings.`;

const HUMAN = `Here is the attempt. Diagnose every question and produce the consolidated guidance.

{attempt_json}`;

export const guidancePrompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM],
  ["human", HUMAN],
]);

/* ------------------------------------------------------------------ *
 * The chain: prompt -> Gemini (JSON-schema constrained) -> zod parse.
 * ------------------------------------------------------------------ */
export function createGuidanceChain({ apiKey, model = DEFAULT_MODEL, thinkingLevel } = {}) {
  const llm = new ChatGoogleGenerativeAI({
    apiKey,
    model,
    // Gemini 3 thinks before answering; "low" trades a little depth for speed.
    ...(thinkingLevel ? { thinkingConfig: { thinkingLevel } } : {}),
    // Fifteen questions with sources is a long answer, and on thinking models
    // the reasoning tokens come out of the same budget.
    maxOutputTokens: 16384,
    // One retry for a momentary blip; an overloaded model is better handled by
    // moving on to the next one (generateGuidance) than by backing off here.
    maxRetries: 1,
  });

  return guidancePrompt
    .pipe(llm.withStructuredOutput(GuidanceResponse, { name: "study_guidance" }))
    // A response that fails schema parsing is worth one more try; the network
    // retries above do not cover that case. Anything else (bad key, quota,
    // abort) is rethrown at once -- throwing here is how p-retry stops early.
    .withRetry({
      stopAfterAttempt: 2,
      onFailedAttempt: (err) => {
        if (err?.lc_error_code !== "OUTPUT_PARSING_FAILURE") throw err;
      },
    });
}

/** Format the variables the prompt expects from a validated request. */
export function promptInput(request) {
  return { attempt_json: JSON.stringify(request, null, 2) };
}

/**
 * Line the response up with the request: input order, one entry per question,
 * nothing Gemini invented.  A question Gemini skipped comes back as null and the
 * UI shows the question without guidance rather than failing the whole page.
 *
 * Topic names are the join key between questions, concepts and resources, so
 * they are matched ignoring case and spacing, and a name Gemini used without
 * listing it becomes a topic with no resources rather than vanishing.
 */
export function alignToRequest(request, response) {
  const known = new Set(request.questions.map((q) => q.questionId));

  const norm = (name) => String(name).trim().toLowerCase().replace(/\s+/g, " ");
  const topics = {};
  const canonical = new Map();
  const topicName = (name, create = true) => {
    const key = norm(name);
    if (!key) return null;
    if (!canonical.has(key)) {
      if (!create) return String(name).trim();
      canonical.set(key, String(name).trim());
      topics[String(name).trim()] = { name: String(name).trim(), resources: [] };
    }
    return canonical.get(key);
  };
  for (const t of response.topics) {
    const name = topicName(t.name);
    if (name && topics[name].resources.length === 0) topics[name].resources = t.resources;
  }

  const byId = new Map(
    response.questions
      .filter((g) => known.has(g.questionId))
      .map((g) => [
        g.questionId,
        { ...g, topics: [...new Set(g.topics.map((n) => topicName(n)).filter(Boolean))] },
      ]),
  );

  return {
    overall: {
      ...response.overall,
      weakConcepts: response.overall.weakConcepts.map((c) => ({
        ...c,
        concept: topicName(c.concept, false),
        questionIds: c.questionIds.filter((id) => known.has(id)),
      })),
      strongConcepts: response.overall.strongConcepts.map((c) => topicName(c, false)),
    },
    topics,
    questions: Object.fromEntries(
      request.questions.map((q) => [q.questionId, byId.get(q.questionId) ?? null]),
    ),
  };
}

/** One chain per model, primary first. */
export function createGuidanceChains({ apiKey, models, thinkingLevel }) {
  return [...new Set(models)].map((model) => ({
    model,
    chain: createGuidanceChain({ apiKey, model, thinkingLevel }),
  }));
}

// Errors that mean "this model, right now" rather than "this request": worth
// trying the next model. A bad key or a cancelled request is not.
export function isModelUnavailable(err) {
  if ([404, 429, 500, 503, 504].includes(err?.status)) return true;
  // A model that does not accept the thinking setting: the next one might.
  if (err?.status === 400 && /thinking/i.test(String(err?.message))) return true;
  if (err?.lc_error_code === "OUTPUT_PARSING_FAILURE") return true;
  return /UNAVAILABLE|high demand|overloaded|RESOURCE_EXHAUSTED/i.test(
    String(err?.message),
  );
}

/**
 * Runs the chains in order until one answers, then checks every resource link
 * (resources.mjs). Returns the guidance and the model that actually wrote it,
 * so the page credits the right one.
 */
export async function generateGuidance(chains, request, { signal, onFallback } = {}) {
  let lastErr;
  for (const { model, chain } of chains) {
    let raw;
    try {
      raw = await chain.invoke(promptInput(request), { signal });
    } catch (err) {
      lastErr = err;
      if (signal?.aborted || !isModelUnavailable(err)) throw err;
      onFallback?.(model, err);
      continue;
    }
    const guidance = alignToRequest(request, raw);
    const { topics, stats } = await checkResourceLinks(guidance.topics, { signal });
    return { guidance: { ...guidance, topics }, model, linkStats: stats };
  }
  throw lastErr;
}

// The SDK's errors embed the whole Google error payload; the results page
// needs one readable sentence.
export function describeFailure(err, models) {
  const msg = String(err?.message || err || "unknown error");
  if (/API_KEY_INVALID|API key not valid/i.test(msg)) {
    return "Gemini rejected the API key. Check GEMINI_API_KEY in .env.local.";
  }
  if (err?.status === 503 || /UNAVAILABLE|high demand|overloaded/i.test(msg)) {
    return `Gemini is overloaded right now (tried ${models.join(", ")}). Try again in a minute.`;
  }
  if (/\b429\b|RESOURCE_EXHAUSTED|quota/i.test(msg)) {
    return "Gemini quota or rate limit reached. Wait a minute and try again.";
  }
  if (/\b404\b/.test(msg)) {
    return `None of ${models.join(", ")} is available to this key. Set GEMINI_MODEL in .env.local.`;
  }
  if (err?.lc_error_code === "OUTPUT_PARSING_FAILURE") {
    return "Gemini's answer did not match the expected format. Try again.";
  }
  return `Gemini request failed: ${msg.replace(/\s*\[\{[\s\S]*$/, "").slice(0, 300)}`;
}
