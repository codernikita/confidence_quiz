/**
 * Seed the Firestore question bank and config from src/data/questions.js.
 *
 *   1. Firebase console > Project settings > Service accounts > Generate key
 *   2. Save it as ./serviceAccountKey.json  (already in .gitignore)
 *   3. npm run seed
 *
 * Re-running is safe: documents are written by id, so it overwrites rather
 * than duplicates.  Pass --wipe to delete questions that are no longer in the
 * bundled bank.
 */

import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { initializeApp, cert } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);
const KEY_PATH =
  process.env.GOOGLE_APPLICATION_CREDENTIALS ||
  join(ROOT, "serviceAccountKey.json");

if (!existsSync(KEY_PATH)) {
  console.error(
    `\nNo service account key at ${KEY_PATH}\n` +
      "Firebase console > Project settings > Service accounts > Generate new private key,\n" +
      "save it there, then re-run `npm run seed`.\n",
  );
  process.exit(1);
}

const serviceAccount = JSON.parse(readFileSync(KEY_PATH, "utf8"));
initializeApp({ credential: cert(serviceAccount) });
const db = getFirestore();

const questionsUrl = pathToFileURL(join(ROOT, "src", "data", "questions.js"));

const { CALIBRATION_QUESTIONS, TECHNICAL_QUESTIONS, DEFAULT_CONFIG } =
  await import(questionsUrl.href);

const wipe = process.argv.includes("--wipe");

async function main() {
  const all = [
    ...CALIBRATION_QUESTIONS.map((q, i) => ({ ...q, order: i })),
    ...TECHNICAL_QUESTIONS.map((q, i) => ({ ...q, order: i })),
  ];

  if (all.filter((q) => q.block === "calibration").length !== 3) {
    console.error("The calibration block must contain exactly 3 questions.");
    process.exit(1);
  }

  const batch = db.batch();
  for (const q of all) {
    const { id, ...data } = q;
    batch.set(db.collection("questions").doc(id), data, { merge: false });
  }
  batch.set(db.collection("settings").doc("config"), DEFAULT_CONFIG, {
    merge: true,
  });
  await batch.commit();
  console.log(`seeded ${all.length} questions + settings/config`);

  if (wipe) {
    const keep = new Set(all.map((q) => q.id));
    const snap = await db.collection("questions").get();
    const stale = snap.docs.filter((d) => !keep.has(d.id));
    if (stale.length) {
      const del = db.batch();
      stale.forEach((d) => del.delete(d.ref));
      await del.commit();
      console.log(`removed ${stale.length} stale question(s)`);
    } else {
      console.log("no stale questions to remove");
    }
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
