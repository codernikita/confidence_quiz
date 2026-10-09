#!/usr/bin/env bash
#
# Deploy the study-guidance API to AWS Lambda behind a public function URL.
#
#   SITE_ORIGIN=https://d3a11a2oec8tq3.cloudfront.net ./scripts/deploy_guidance_lambda.sh
#   ./scripts/deploy_guidance_lambda.sh --print-policy   # IAM policy the deploy user needs
#
# Safe to re-run: the first run creates the role, function and URL; later runs
# update code and configuration. Afterwards it writes VITE_GUIDANCE_API_URL into
# .env.production, so the next ./scripts/deploy_aws.sh builds the site against it.
#
# Why a function URL and not CloudFront /api/*: a guidance call takes 40-120 s,
# and CloudFront gives up on an origin after 60 s at most. Function URLs allow
# the full Lambda timeout.
#
# GEMINI_API_KEY and the Firebase project id are read from .env.local /
# .env.production unless already exported. The key is never printed.

set -euo pipefail

FUNCTION="${FUNCTION:-bis-quiz-guidance}"
ROLE_NAME="${ROLE_NAME:-bis-quiz-guidance-role}"
REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
REGION="${REGION:-ap-south-1}"
TIMEOUT=180
MEMORY=512
LOG_RETENTION_DAYS=14
PROFILE_ARG=()
[ -n "${AWS_PROFILE:-}" ] && PROFILE_ARG=(--profile "$AWS_PROFILE")
# ${arr[@]+...} because macOS's bash 3.2 calls an empty array unbound under set -u.
aws_() { aws ${PROFILE_ARG[@]+"${PROFILE_ARG[@]}"} --region "$REGION" "$@"; }

# "Does it exist?" for a get-* call: 0 = yes, 1 = no, exits on access denied
# rather than mistaking a missing permission for a missing resource.
exists() {
  local err
  if err="$(aws_ "$@" 2>&1 >/dev/null)"; then return 0; fi
  if grep -qE "AccessDenied|not authorized" <<<"$err"; then
    echo "error: this AWS identity is not allowed to manage the guidance function." >&2
    echo "$err" | head -2 >&2
    echo "An admin needs to attach the policy printed by:" >&2
    echo "  ./scripts/deploy_guidance_lambda.sh --print-policy" >&2
    exit 1
  fi
  return 1
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACCOUNT_ID="$(aws_ sts get-caller-identity --query Account --output text)"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION}"
LOG_GROUP="/aws/lambda/${FUNCTION}"

if [ "${1:-}" = "--print-policy" ]; then
  cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ManageGuidanceFunction",
      "Effect": "Allow",
      "Action": [
        "lambda:GetFunction",
        "lambda:GetFunctionConfiguration",
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:GetFunctionUrlConfig",
        "lambda:CreateFunctionUrlConfig",
        "lambda:UpdateFunctionUrlConfig",
        "lambda:AddPermission",
        "lambda:GetPolicy",
        "lambda:PutFunctionConcurrency",
        "lambda:DeleteFunctionConcurrency"
      ],
      "Resource": "${FUNCTION_ARN}"
    },
    {
      "Sid": "ManageGuidanceRole",
      "Effect": "Allow",
      "Action": ["iam:GetRole", "iam:CreateRole"],
      "Resource": "${ROLE_ARN}"
    },
    {
      "Sid": "AttachOnlyBasicLogging",
      "Effect": "Allow",
      "Action": "iam:AttachRolePolicy",
      "Resource": "${ROLE_ARN}",
      "Condition": {
        "ArnEquals": {
          "iam:PolicyARN": "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        }
      }
    },
    {
      "Sid": "PassRoleToLambdaOnly",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${ROLE_ARN}",
      "Condition": { "StringEquals": { "iam:PassedToService": "lambda.amazonaws.com" } }
    },
    {
      "Sid": "GuidanceLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:PutRetentionPolicy",
        "logs:FilterLogEvents",
        "logs:GetLogEvents",
        "logs:StartLiveTail"
      ],
      "Resource": [
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:${LOG_GROUP}",
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:${LOG_GROUP}:*"
      ]
    }
  ]
}
JSON
  exit 0
fi

SITE_ORIGIN="${SITE_ORIGIN:-}"
if [ -z "$SITE_ORIGIN" ]; then
  echo "SITE_ORIGIN is not set. Example:" >&2
  echo "  SITE_ORIGIN=https://d3a11a2oec8tq3.cloudfront.net ./scripts/deploy_guidance_lambda.sh" >&2
  exit 1
fi
SITE_ORIGIN="${SITE_ORIGIN%/}"

# --- configuration from the env files (shell exports win) -------------------
# Written by node so the key is JSON-escaped correctly and never echoed.
mkdir -p .lambda
ENV_JSON="$(mktemp -t bisq-lambda-env)"  # private temp file, outside the project
umask 077
node --input-type=module - "$ENV_JSON" <<'NODE'
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseEnv } from "node:util";
const files = {};
for (const f of [".env", ".env.production", ".env.local"]) {
  if (existsSync(f)) Object.assign(files, parseEnv(readFileSync(f, "utf8")));
}
const get = (k) => String(process.env[k] ?? files[k] ?? "").trim();
const vars = {
  GEMINI_API_KEY: get("GEMINI_API_KEY") || get("GOOGLE_API_KEY"),
  FIREBASE_PROJECT_ID: get("FIREBASE_PROJECT_ID") || get("VITE_FB_PROJECT_ID"),
};
for (const k of ["GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "GEMINI_THINKING_LEVEL"]) {
  if (process.env[k] !== undefined || files[k] !== undefined) vars[k] = get(k);
}
if (!vars.GEMINI_API_KEY) { console.error("GEMINI_API_KEY not found in the environment or .env.local"); process.exit(1); }
if (!vars.FIREBASE_PROJECT_ID) { console.error("No Firebase project id (VITE_FB_PROJECT_ID). Refusing to deploy an unauthenticated public endpoint."); process.exit(1); }
writeFileSync(process.argv[2], JSON.stringify({ Variables: vars }));
console.log(`config: firebase project ${vars.FIREBASE_PROJECT_ID}, key ${vars.GEMINI_API_KEY.length} chars, overrides: ${Object.keys(vars).filter((k) => k.startsWith("GEMINI_") && k !== "GEMINI_API_KEY").join(", ") || "none"}`);
NODE
umask 022
trap 'rm -f "$ENV_JSON"' EXIT

echo "==> bundling"
node scripts/build_lambda.mjs

# --- execution role ---------------------------------------------------------
if ! exists iam get-role --role-name "$ROLE_NAME"; then
  echo "==> creating role $ROLE_NAME"
  aws_ iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    >/dev/null
  aws_ iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  NEW_ROLE=1
fi

# --- log group with a retention period (the default keeps logs forever) -----
aws_ logs create-log-group --log-group-name "$LOG_GROUP" 2>/dev/null || true
aws_ logs put-retention-policy --log-group-name "$LOG_GROUP" \
  --retention-in-days "$LOG_RETENTION_DAYS" 2>/dev/null ||
  echo "    (could not set log retention; continuing)"

# --- function ---------------------------------------------------------------
if exists lambda get-function --function-name "$FUNCTION"; then
  echo "==> updating function $FUNCTION"
  aws_ lambda update-function-code --function-name "$FUNCTION" \
    --zip-file fileb://.lambda/function.zip >/dev/null
  aws_ lambda wait function-updated-v2 --function-name "$FUNCTION"
  aws_ lambda update-function-configuration --function-name "$FUNCTION" \
    --timeout "$TIMEOUT" --memory-size "$MEMORY" \
    --environment "file://$ENV_JSON" >/dev/null
  aws_ lambda wait function-updated-v2 --function-name "$FUNCTION"
else
  echo "==> creating function $FUNCTION"
  # A brand-new role takes a few seconds to become assumable by Lambda.
  for attempt in 1 2 3 4 5 6 7 8; do
    if aws_ lambda create-function --function-name "$FUNCTION" \
      --runtime nodejs22.x --architectures arm64 \
      --handler index.handler --role "$ROLE_ARN" \
      --zip-file fileb://.lambda/function.zip \
      --timeout "$TIMEOUT" --memory-size "$MEMORY" \
      --environment "file://$ENV_JSON" >/dev/null 2>.lambda/create.err; then
      break
    fi
    if [ "${NEW_ROLE:-0}" = 1 ] && grep -q "cannot be assumed" .lambda/create.err && [ "$attempt" -lt 8 ]; then
      sleep 5
      continue
    fi
    cat .lambda/create.err >&2
    exit 1
  done
  aws_ lambda wait function-active-v2 --function-name "$FUNCTION"
fi

if [ -n "${RESERVED_CONCURRENCY:-}" ]; then
  aws_ lambda put-function-concurrency --function-name "$FUNCTION" \
    --reserved-concurrent-executions "$RESERVED_CONCURRENCY" >/dev/null ||
    echo "    (could not reserve concurrency -- new accounts often cannot; continuing)"
fi

# --- function URL: public, CORS limited to the site ------------------------
CORS="{\"AllowOrigins\":[\"${SITE_ORIGIN}\"],\"AllowMethods\":[\"POST\"],\"AllowHeaders\":[\"content-type\",\"authorization\"],\"MaxAge\":86400}"
if exists lambda get-function-url-config --function-name "$FUNCTION"; then
  aws_ lambda update-function-url-config --function-name "$FUNCTION" --cors "$CORS" >/dev/null
else
  echo "==> creating function URL"
  aws_ lambda create-function-url-config --function-name "$FUNCTION" \
    --auth-type NONE --cors "$CORS" >/dev/null
fi

# A public URL needs both statements. Authentication happens in the handler,
# against the student's Firebase ID token.
aws_ lambda add-permission --function-name "$FUNCTION" \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl --principal "*" \
  --function-url-auth-type NONE >/dev/null 2>&1 || true
aws_ lambda add-permission --function-name "$FUNCTION" \
  --statement-id FunctionURLInvokeAllowPublicAccess \
  --action lambda:InvokeFunction --principal "*" \
  --invoked-via-function-url >/dev/null 2>&1 || true

URL="$(aws_ lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text)"
API_URL="${URL%/}/api/guidance"

echo "==> smoke test"
curl -fsS --max-time 60 "${URL%/}/api/health" && echo

# --- point the production build at it ---------------------------------------
touch .env.production
if grep -q '^VITE_GUIDANCE_API_URL=' .env.production; then
  sed -i.bak "s#^VITE_GUIDANCE_API_URL=.*#VITE_GUIDANCE_API_URL=${API_URL}#" .env.production
  rm -f .env.production.bak
else
  printf '\n# Study-guidance API (scripts/deploy_guidance_lambda.sh)\nVITE_GUIDANCE_API_URL=%s\n' "$API_URL" >>.env.production
fi

echo
echo "guidance API: $API_URL"
echo "written to .env.production. Now rebuild and publish the site:"
echo "  BUCKET=<bucket> DIST_ID=<distribution id> ./scripts/deploy_aws.sh"
