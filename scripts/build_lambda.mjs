/**
 * Bundles server/lambda.mjs into one ESM file and zips it for Lambda.
 *
 *   node scripts/build_lambda.mjs   ->  .lambda/function.zip
 *
 * One bundled file instead of shipping node_modules: the zip stays a few MB,
 * and only the code the handler actually imports is deployed.
 */
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const OUT = join(ROOT, ".lambda");

// Remove only this script's outputs: the deploy script keeps other files here.
mkdirSync(OUT, { recursive: true });
for (const f of ["index.mjs", "function.zip"]) rmSync(join(OUT, f), { force: true });

await build({
  entryPoints: [join(ROOT, "server/lambda.mjs")],
  outfile: join(OUT, "index.mjs"),
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node22",
  minify: true,
  sourcemap: false,
  legalComments: "none",
  // Some CommonJS dependencies call require() at runtime; give the ESM bundle one.
  banner: {
    js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);",
  },
  logLevel: "warning",
});

execFileSync("zip", ["-q", "-j", join(OUT, "function.zip"), join(OUT, "index.mjs")]);
const kb = (f) => `${Math.round(statSync(f).size / 1024)} KB`;
console.log(`built .lambda/index.mjs (${kb(join(OUT, "index.mjs"))}), .lambda/function.zip (${kb(join(OUT, "function.zip"))})`);
