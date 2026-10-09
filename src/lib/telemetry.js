/**
 * Per-question behavioural telemetry.
 *
 * Only counts time the tab is actually visible and the window focused -- a
 * student who walks away mid-question would otherwise look like deep
 * deliberation, and `time_spent` is one of the model's inputs.
 *
 * Fields mirror the existing Bis-quiz schema exactly:
 *   timeSpent · optionChanges · markedForReview · reviewClickCount · isCorrect
 */

export function createTelemetry() {
  const store = new Map(); // questionId -> record
  let activeId = null;
  let activeSince = null;
  let paused = false;

  const blank = () => ({
    timeSpent: 0,
    optionChanges: 0,
    reviewClickCount: 0,
    markedForReview: false,
    visits: 0,
    firstSeenAt: null,
    selectedOption: "",
    confidenceRating: 0,
  });

  const rec = (id) => {
    if (!store.has(id)) store.set(id, blank());
    return store.get(id);
  };

  const flush = () => {
    if (activeId === null || activeSince === null) return;
    const elapsed = (Date.now() - activeSince) / 1000;
    if (elapsed > 0 && elapsed < 3600) rec(activeId).timeSpent += elapsed;
    activeSince = paused ? null : Date.now();
  };

  const api = {
    enter(id) {
      if (activeId === id) return;
      flush();
      activeId = id;
      activeSince = paused ? null : Date.now();
      const r = rec(id);
      r.visits += 1;
      if (r.firstSeenAt === null) r.firstSeenAt = new Date().toISOString();
    },

    /** Call when the tab hides or the window blurs. */
    pause() {
      if (paused) return;
      flush();
      paused = true;
      activeSince = null;
    },

    resume() {
      if (!paused) return;
      paused = false;
      activeSince = activeId === null ? null : Date.now();
    },

    /** An option change is a change -- the first selection does not count. */
    select(id, value) {
      const r = rec(id);
      if (r.selectedOption !== "" && r.selectedOption !== value) {
        r.optionChanges += 1;
      }
      r.selectedOption = value;
    },

    toggleReview(id) {
      const r = rec(id);
      r.markedForReview = !r.markedForReview;
      r.reviewClickCount += 1;
      return r.markedForReview;
    },

    setConfidence(id, rating) {
      rec(id).confidenceRating = Number(rating) || 0;
    },

    snapshot(id) {
      const r = rec(id);
      // Include the time accruing right now so the UI can show a live clock.
      const live =
        activeId === id && activeSince !== null
          ? (Date.now() - activeSince) / 1000
          : 0;
      return { ...r, timeSpent: r.timeSpent + live };
    },

    /** Freeze everything and return one row per question, in bank order. */
    finalise(questions) {
      flush();
      activeId = null;
      activeSince = null;
      return questions.map((q) => {
        const r = rec(q.id);
        const selected = r.selectedOption || "";
        return {
          questionId: q.id,
          block: q.block,
          type: q.type || "",
          difficulty: q.difficulty || "medium",
          text: q.text,
          prompt: q.prompt || "",
          options: q.options,
          correctAnswer: q.correctAnswer,
          explanation: q.explanation || "",
          selectedOption: selected,
          isCorrect: Boolean(selected) && selected === q.correctAnswer,
          timeSpent: Math.round(r.timeSpent * 10) / 10,
          optionChanges: r.optionChanges,
          markedForReview: Boolean(r.markedForReview),
          reviewClickCount: r.reviewClickCount,
          visits: r.visits,
          confidenceRating: r.confidenceRating,
          firstSeenAt: r.firstSeenAt,
        };
      });
    },
  };

  return api;
}
