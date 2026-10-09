/**
 * resources.mjs -- checks every study-resource link before a student sees it.
 *
 * Gemini is good at choosing resources and bad at their addresses: measured on
 * this project, every one of 8 YouTube links it produced from memory was an
 * invented video ID (404), and 3 of 4 article links were dead, while the
 * titles and channels were real. Search grounding would fix that, but it needs
 * a billed plan. So each link is checked here:
 *
 *   YouTube   -> YouTube's public oEmbed endpoint: confirms the video exists
 *                and returns its real title and channel (no API key needed)
 *   other     -> fetched: must answer 2xx with a page that is not a 404 page
 *
 * A link that passes, and whose real title is about the topic, is shown as a
 * direct link. Anything else becomes a search the student can trust: YouTube
 * search for "title channel" for videos, a site-restricted Google search for
 * everything else. Nothing unverified is ever presented as a direct link.
 *
 * The URLs come from model output, so fetching them is an SSRF surface: only
 * http(s) on default ports to hostnames that resolve to public addresses, and
 * every redirect hop is re-checked.
 */
import dns from "node:dns/promises";
import net from "node:net";

const CHECK_TIMEOUT_MS = 5_000; // per link
const BUDGET_MS = 15_000; // all links together; leftovers fall back to search
const CONCURRENCY = 10;
const MAX_REDIRECTS = 4;
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const USER_AGENT =
  "Mozilla/5.0 (compatible; BISQuizLinkCheck/1.0; checks study links before showing them)";

/* ------------------------------------------------------------------ *
 * Public-address guard.
 * ------------------------------------------------------------------ */
const PRIVATE = new net.BlockList();
for (const [addr, prefix] of [
  ["0.0.0.0", 8],
  ["10.0.0.0", 8],
  ["100.64.0.0", 10],
  ["127.0.0.0", 8],
  ["169.254.0.0", 16],
  ["172.16.0.0", 12],
  ["192.0.0.0", 24],
  ["192.168.0.0", 16],
  ["198.18.0.0", 15],
  ["224.0.0.0", 3],
]) {
  PRIVATE.addSubnet(addr, prefix, "ipv4");
}
for (const [addr, prefix] of [
  ["::", 127], // :: and ::1
  ["fc00::", 7],
  ["fe80::", 10],
  ["ff00::", 8],
  ["64:ff9b::", 96],
]) {
  PRIVATE.addSubnet(addr, prefix, "ipv6");
}

async function isPublicUrl(url) {
  if (url.protocol !== "https:" && url.protocol !== "http:") return false;
  if (url.port && url.port !== "80" && url.port !== "443") return false;
  if (url.username || url.password) return false;
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (net.isIP(host) || !host.includes(".") || /\.(local|internal|localhost)$/i.test(host)) {
    return false;
  }
  try {
    const addrs = await dns.lookup(host, { all: true });
    return (
      addrs.length > 0 &&
      addrs.every((a) => !PRIVATE.check(a.address, a.family === 6 ? "ipv6" : "ipv4"))
    );
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------------ *
 * Checks.
 * ------------------------------------------------------------------ */
const decode = (s) =>
  s
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#0?39;|&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();

async function readTitle(res) {
  if (!/html/i.test(res.headers.get("content-type") || "")) return "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let html = "";
  try {
    while (html.length < 150_000) {
      const { done, value } = await reader.read();
      if (done) break;
      html += decoder.decode(value, { stream: true });
      const m = html.match(/<title[^>]*>([^<]*)<\/title>/i);
      if (m) return decode(m[1]);
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  return "";
}

const SOFT_404 = /\b(404|not found|page (?:does not|doesn't|can't|cannot) (?:exist|be found)|no longer available)\b/i;

async function checkPage(rawUrl, signal) {
  let url = new URL(rawUrl);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    if (!(await isPublicUrl(url))) return { ok: false, reason: "blocked" };
    const res = await fetch(url, {
      redirect: "manual",
      headers: { "user-agent": USER_AGENT, accept: "text/html,*/*;q=0.8" },
      signal: AbortSignal.any([signal, AbortSignal.timeout(CHECK_TIMEOUT_MS)]),
    });
    const location = res.headers.get("location");
    if (res.status >= 300 && res.status < 400 && location) {
      res.body?.cancel().catch(() => {});
      url = new URL(location, url);
      continue;
    }
    if (!res.ok) {
      res.body?.cancel().catch(() => {});
      return { ok: false, reason: `HTTP ${res.status}` };
    }
    const title = await readTitle(res);
    if (SOFT_404.test(title)) return { ok: false, reason: "not-found page" };
    // A deep link that lands on the home page is how many sites say "missing".
    const isHome = (u) => u.pathname.replace(/\/+$/, "") === "";
    if (hop > 0 && isHome(url) && !isHome(new URL(rawUrl))) {
      return { ok: false, reason: "redirected to home page" };
    }
    return { ok: true, title, finalUrl: url.href };
  }
  return { ok: false, reason: "too many redirects" };
}

const YT_HOSTS = /^(?:www\.|m\.|music\.)?youtube\.com$|^youtu\.be$/i;

/** Canonical watch/playlist URL for a YouTube link, or null if it is not one. */
export function youtubeCanonical(rawUrl) {
  let u;
  try {
    u = new URL(rawUrl);
  } catch {
    return null;
  }
  if (!YT_HOSTS.test(u.hostname)) return null;
  const id =
    u.hostname === "youtu.be"
      ? u.pathname.slice(1)
      : u.searchParams.get("v") || u.pathname.match(/^\/(?:embed|shorts|live)\/([^/?#]+)/)?.[1];
  if (id && /^[\w-]{11}$/.test(id)) return `https://www.youtube.com/watch?v=${id}`;
  const list = u.searchParams.get("list");
  if (list && /^[\w-]+$/.test(list)) return `https://www.youtube.com/playlist?list=${list}`;
  return null;
}

async function checkYouTube(canonical, signal) {
  const res = await fetch(
    `https://www.youtube.com/oembed?format=json&url=${encodeURIComponent(canonical)}`,
    { signal: AbortSignal.any([signal, AbortSignal.timeout(CHECK_TIMEOUT_MS)]) },
  );
  if (!res.ok) return { ok: false, reason: `oEmbed ${res.status}` };
  const j = await res.json();
  return { ok: true, title: decode(j.title || ""), author: decode(j.author_name || ""), finalUrl: canonical };
}

/* ------------------------------------------------------------------ *
 * Relevance: an ID that exists can still be the wrong video. The real
 * title must share words with what Gemini said it was, or with the topic.
 * ------------------------------------------------------------------ */
const STOP = new Set(
  "the and for with from what how why are you your this that into its when then than about using use explained explanation explain tutorial tutorials video videos part lecture lesson introduction intro guide beginners beginner complete full course learn learning easy simple minutes min".split(
    " ",
  ),
);
// Crude stemming is enough here: "syllogisms" / "syllogistic" -> "syllog",
// "normalization" / "normal" -> "normal", "joins" -> "join".
const stem = (w) => {
  const s = w.length > 3 && w.endsWith("s") && !w.endsWith("ss") ? w.slice(0, -1) : w;
  return s.length > 6 ? s.slice(0, 6) : s;
};
const words = (s) =>
  new Set(
    String(s)
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => (w.length >= 3 || /\d/.test(w)) && !STOP.has(w))
      .map(stem),
  );

/**
 * Videos need two shared words: a guessed YouTube ID that happens to exist is
 * usually some unrelated video, and its title is all there is to judge by.
 * Pages need one, counting the words in their URL path too -- a URL Gemini
 * chose for this topic that really exists is already strong evidence.
 */
export function looksRelevant({ foundTitle, foundUrl = "", claimedTitle, topicName, isVideo }) {
  let path = "";
  if (!isVideo && foundUrl) {
    try {
      path = decodeURIComponent(new URL(foundUrl).pathname);
    } catch {
      /* ignore */
    }
  }
  const found = words(`${foundTitle} ${path}`);
  if (found.size === 0) return true; // nothing to judge by; the page exists
  const expected = new Set([...words(claimedTitle), ...words(topicName)]);
  let shared = 0;
  for (const w of found) if (expected.has(w)) shared++;
  return shared >= (isVideo ? Math.min(2, found.size) : 1);
}

/* ------------------------------------------------------------------ *
 * Search fallbacks -- always real, and labelled as searches in the UI.
 * ------------------------------------------------------------------ */
function searchLink(resource, isVideo) {
  if (isVideo) {
    const q = `${resource.title} ${resource.creator}`.trim();
    return `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
  }
  let site = "";
  try {
    const u = new URL(resource.url);
    const host = u.hostname.replace(/^\[|\]$/g, "");
    if (/^https?:$/.test(u.protocol) && host.includes(".") && !net.isIP(host)) {
      site = host.replace(/^www\./, "");
    }
  } catch {
    /* no usable URL: search by title and creator only */
  }
  const q = site ? `site:${site} ${resource.title}` : `${resource.title} ${resource.creator}`;
  return `https://www.google.com/search?q=${encodeURIComponent(q.trim())}`;
}

/* ------------------------------------------------------------------ *
 * Entry point.
 * ------------------------------------------------------------------ */
const cache = new Map(); // canonical url -> { at, promise }

function cachedCheck(key, run) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.promise;
  if (cache.size > 5000) cache.clear();
  const promise = run().catch((err) => ({ ok: false, reason: err?.name || "error", transient: true }));
  cache.set(key, { at: Date.now(), promise });
  // Do not remember timeouts and network blips as "dead".
  promise.then((r) => r.transient && cache.delete(key));
  return promise;
}

async function resolveOne(resource, topicName, signal) {
  const url = String(resource.url || "").trim();
  const yt = url ? youtubeCanonical(url) : null;
  const isVideo = resource.kind === "video" || Boolean(yt);
  const fallback = {
    ...resource,
    href: searchLink(resource, isVideo),
    link: isVideo ? "youtube-search" : "web-search",
  };
  if (!url || signal.aborted) return fallback;

  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return fallback;
  }

  const result = yt
    ? await cachedCheck(yt, () => checkYouTube(yt, signal))
    : await cachedCheck(parsed.href, () => checkPage(parsed.href, signal));

  // A site's home page is not "a resource on this topic" -- a site-restricted
  // search for the title gets closer. A free course's home page is the
  // exception: that is the resource.
  const landsOnHome =
    !yt && result.ok && resource.kind !== "course" && new URL(result.finalUrl).pathname.replace(/\/+$/, "") === "";
  const relevant =
    result.ok &&
    !landsOnHome &&
    looksRelevant({
      foundTitle: result.title,
      foundUrl: result.finalUrl,
      claimedTitle: resource.title,
      topicName,
      isVideo: Boolean(yt),
    });
  if (!relevant) return fallback;

  return {
    ...resource,
    // For YouTube the real title and channel are known; show them rather
    // than Gemini's recollection of them.
    ...(yt ? { title: result.title || resource.title, creator: result.author || resource.creator } : {}),
    href: result.finalUrl,
    link: "verified",
  };
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  });
  await Promise.all(workers);
  return out;
}

/**
 * Resolve every resource of every topic to a link that is safe to show.
 * Never throws: a check that fails or runs out of time becomes a search link.
 */
export async function checkResourceLinks(topics, { signal } = {}) {
  const budget = AbortSignal.timeout(BUDGET_MS);
  const combined = signal ? AbortSignal.any([signal, budget]) : budget;

  const jobs = Object.values(topics).flatMap((t) =>
    t.resources.map((r, i) => ({ topic: t.name, index: i, resource: r })),
  );
  const resolved = await mapLimit(jobs, CONCURRENCY, (job) =>
    resolveOne(job.resource, job.topic, combined).catch(() => ({
      ...job.resource,
      href: searchLink(job.resource, job.resource.kind === "video"),
      link: job.resource.kind === "video" ? "youtube-search" : "web-search",
    })),
  );

  const out = Object.fromEntries(
    Object.values(topics).map((t) => [t.name, { ...t, resources: [...t.resources] }]),
  );
  jobs.forEach((job, k) => {
    out[job.topic].resources[job.index] = resolved[k];
  });

  const stats = { total: resolved.length, verified: 0, youtubeSearch: 0, webSearch: 0 };
  for (const r of resolved) {
    if (r.link === "verified") stats.verified++;
    else if (r.link === "youtube-search") stats.youtubeSearch++;
    else stats.webSearch++;
  }
  return { topics: out, stats };
}
