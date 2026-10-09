# Deploying to AWS

The app is a static bundle plus Firestore. Nothing runs on a server, so AWS only
has to serve files: **S3 for storage, CloudFront for HTTPS and the CDN.**

Everything below is copy-pasteable. Steps 0–7 are the one-time setup; after that
every deploy is step 8 alone.

> **zsh note.** Comments are kept on their own lines throughout, never trailing a
> command. macOS zsh does not enable `interactive_comments` by default, so a
> pasted `somecmd  # note` sends `#` and the note along as arguments — which is
> how `firebase use --add  # pick the project` becomes *"Too many arguments"*.
> Run `setopt interactive_comments` if you want trailing comments to work.

---

## 0. Prerequisites

```bash
# need aws v2.x, node 18+, and working credentials
aws --version
node --version
aws sts get-caller-identity
```

If `aws` is missing:

```bash
# macOS
brew install awscli
# then
# asks for: access key, secret, region (e.g. ap-south-1), output format json
aws configure
```

Pick names once and export them — every later command reuses these:

```bash
export AWS_REGION=ap-south-1
# BUCKET must be globally unique across all of AWS
export BUCKET=bis-quiz-web-$(date +%s)
export PROJECT=bis-quiz
echo "BUCKET=$BUCKET"
```

---

## 1. Set up Firestore (the only backend)

In the [Firebase console](https://console.firebase.google.com):

1. **Create a project** (or reuse an existing one).
2. **Build → Firestore Database → Create database → Production mode.**
3. **Build → Authentication → Get started → Sign-in method → Google → Enable.**
   You will be asked for a **public-facing project name** (students see this on
   the Google consent screen — use something they will recognise, e.g.
   "Thapar Confidence Quiz") and a **support email**. Save.

   Google sign-in gives each student a stable identity, which is what lets the
   app stamp `attemptNumber` on repeat sittings. Without a provider enabled,
   every write is rejected.
4. **Project settings → General → Your apps → Web (`</>`)** → register the app
   and copy the config values.

Write those into a build-time env file:

```bash
cd BIS-quiz
cat > .env.production <<'EOF'
VITE_FB_API_KEY=AIza...
VITE_FB_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FB_PROJECT_ID=your-project
VITE_FB_STORAGE_BUCKET=your-project.firebasestorage.app
VITE_FB_MESSAGING_SENDER_ID=000000000000
VITE_FB_APP_ID=1:000000000000:web:abcdef
VITE_USE_FIRESTORE_QUESTIONS=true

# Optional: restrict sign-in to institutional addresses. Empty = any Google
# account. Comma-separated for several domains.
VITE_ALLOWED_EMAIL_DOMAINS=
EOF
```

These are public client identifiers by design — they end up in the JS bundle and
that is fine. Access control lives entirely in `firestore.rules`.

---

## 2. Publish the security rules

```bash
npm install -g firebase-tools
firebase login
# firebase use --add is interactive: pick the project, then alias it "default"
firebase use --add
firebase deploy --only firestore:rules
```

`firestore.rules` makes attempts **write-once**, readable only by the student
who created them, and never updatable or deletable from the browser; the
question bank is read-only. Deploy them before the site goes live — Firestore's
default production rules deny everything, and the default *test* rules let
anyone on the internet read every attempt.

---

## 3. Seed the question bank

```bash
# Firebase console → Project settings → Service accounts → Generate new private key
# save it as BIS-quiz/serviceAccountKey.json   (already gitignored)
npm install
npm run seed
```

Seeds `questions/*` (3 calibration + 12 technical) and `settings/config`.
Verify with `npm run check`.
Re-running overwrites by id rather than duplicating. Add `-- --wipe` to also
delete questions no longer in `src/data/questions.js`.

Skipping this step is survivable: the app falls back to the bundled bank in
`src/data/questions.js`.

---

## 4. Create the S3 bucket

The bucket stays **fully private** — CloudFront reads it through an Origin
Access Control. Do not enable S3 static website hosting; that mode requires a
public bucket and only speaks HTTP.

```bash
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"

aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

> `us-east-1` is the exception: it rejects `--create-bucket-configuration`.
> Drop that flag if you deploy there.

---

## 5. Build and upload

```bash
npm ci
npm run build

aws s3 sync dist/ "s3://$BUCKET/" --delete \
  --exclude "index.html" \
  --cache-control "public,max-age=31536000,immutable"

aws s3 cp dist/index.html "s3://$BUCKET/index.html" \
  --cache-control "no-cache,no-store,must-revalidate" \
  --content-type "text/html; charset=utf-8"
```

Two different cache policies on purpose: asset filenames carry a content hash so
they can cache forever, but `index.html` names those assets and must never be
cached, or browsers keep loading the previous build after a deploy.

---

## 6. Create the CloudFront distribution

**6a. Origin Access Control** — lets CloudFront, and only CloudFront, read the
private bucket:

```bash
export OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config "{
    \"Name\": \"$PROJECT-oac\",
    \"Description\": \"OAC for $BUCKET\",
    \"SigningProtocol\": \"sigv4\",
    \"SigningBehavior\": \"always\",
    \"OriginAccessControlOriginType\": \"s3\"
  }" --query 'OriginAccessControl.Id' --output text)
echo "OAC_ID=$OAC_ID"
```

**6b. The distribution.** `CustomErrorResponses` maps 403 and 404 to
`/index.html` with a 200 — a single-page app has no `/results` object in S3, so
without this a refresh on any deep link returns S3's XML error page.

```bash
cat > /tmp/cf-config.json <<JSON
{
  "CallerReference": "$PROJECT-$(date +%s)",
  "Comment": "$PROJECT confidence quiz",
  "Enabled": true,
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 1,
    "Items": [{
      "Id": "s3-$BUCKET",
      "DomainName": "$BUCKET.s3.$AWS_REGION.amazonaws.com",
      "OriginAccessControlId": "$OAC_ID",
      "S3OriginConfig": { "OriginAccessIdentity": "" }
    }]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "s3-$BUCKET",
    "ViewerProtocolPolicy": "redirect-to-https",
    "Compress": true,
    "AllowedMethods": {
      "Quantity": 2,
      "Items": ["GET", "HEAD"],
      "CachedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"] }
    },
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6"
  },
  "CustomErrorResponses": {
    "Quantity": 2,
    "Items": [
      { "ErrorCode": 403, "ResponseCode": "200", "ResponsePagePath": "/index.html", "ErrorCachingMinTTL": 0 },
      { "ErrorCode": 404, "ResponseCode": "200", "ResponsePagePath": "/index.html", "ErrorCachingMinTTL": 0 }
    ]
  },
  "PriceClass": "PriceClass_All"
}
JSON

export DIST_ID=$(aws cloudfront create-distribution \
  --distribution-config file:///tmp/cf-config.json \
  --query 'Distribution.Id' --output text)

export DIST_DOMAIN=$(aws cloudfront get-distribution \
  --id "$DIST_ID" --query 'Distribution.DomainName' --output text)

echo "DIST_ID=$DIST_ID"
echo "URL=https://$DIST_DOMAIN"
```

`658327ea-...` is AWS's managed **CachingOptimized** policy — a fixed, public id,
the same in every account.

**6c. Let that distribution read the bucket:**

```bash
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cat > /tmp/bucket-policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowCloudFrontRead",
    "Effect": "Allow",
    "Principal": { "Service": "cloudfront.amazonaws.com" },
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$BUCKET/*",
    "Condition": {
      "StringEquals": {
        "AWS:SourceArn": "arn:aws:cloudfront::$ACCOUNT_ID:distribution/$DIST_ID"
      }
    }
  }]
}
JSON

aws s3api put-bucket-policy --bucket "$BUCKET" --policy file:///tmp/bucket-policy.json
```

**6d. Wait for it to roll out** (5–15 minutes on the first deploy):

```bash
aws cloudfront wait distribution-deployed --id "$DIST_ID"
echo "live: https://$DIST_DOMAIN"
```

---

## 7. Authorise the domain in Firebase

**Do not skip this.** Google sign-in refuses to run on an origin Firebase does
not know about, so the deployed site will show `auth/unauthorized-domain` and
nobody can get past the sign-in screen — even though it worked perfectly on
`localhost`, which is authorised by default.

Firebase console → **Authentication → Settings → Authorized domains → Add
domain** → paste `$DIST_DOMAIN` (and your custom domain, if any).

Then verify a real end-to-end attempt:

```bash
open "https://$DIST_DOMAIN"
```

Take the quiz once, then confirm the write landed:

```bash
npm run check
```

`quiz_attempts` should go from 0 to 1, and every integrity check should read OK.

---

## 8. Every deploy after this

```bash
cd BIS-quiz
BUCKET=$BUCKET DIST_ID=$DIST_ID ./scripts/deploy_aws.sh
```

That script rebuilds, uploads with the right cache headers, invalidates
CloudFront and waits for the invalidation to finish. Make it executable once:

```bash
chmod +x scripts/deploy_aws.sh
```

---

## 9. The study-guidance API (Gemini, on Lambda)

The results page asks a small API for Gemini's study guidance. It holds the
Gemini key, so it cannot live in the static bundle. It runs on **AWS Lambda
behind a function URL**, not behind CloudFront: a guidance call takes 40–120 s
and CloudFront gives up on an origin after 60 s at most.

The function URL is public, so the handler only serves students signed in to
this quiz's Firebase project: the browser sends its Firebase ID token and the
function verifies it (`server/auth.mjs`). The deploy script refuses to run
without a Firebase project id.

**9a. Permissions (once, by an account admin).** The deploy user needs to
manage one function, one role and one log group:

```bash
./scripts/deploy_guidance_lambda.sh --print-policy > guidance-policy.json
```

An admin attaches it to the deploy user, either in the console (**IAM → Users →
the deploy user → Add permissions → Create inline policy → JSON**) or with:

```bash
aws iam put-user-policy --profile <admin-profile> \
  --user-name lakksh_bisDeployed \
  --policy-name bis-quiz-guidance-deploy \
  --policy-document file://guidance-policy.json
```

**9b. Deploy the API.** Reads `GEMINI_API_KEY` from `.env.local` (never
printed), creates or updates the function, sets CORS on the URL to the site's
origin only, smoke-tests it, and writes `VITE_GUIDANCE_API_URL` into
`.env.production`:

```bash
SITE_ORIGIN=https://$DIST_DOMAIN ./scripts/deploy_guidance_lambda.sh
```

**9c. Rebuild the site** so it knows the API's address (step 8). Without
`VITE_GUIDANCE_API_URL` a production build simply leaves guidance out.

Re-run 9b after changing anything in `server/` or rotating the key. Logs:

```bash
aws logs tail /aws/lambda/bis-quiz-guidance --follow
```

Optional: `RESERVED_CONCURRENCY=5` on 9b caps simultaneous calls. New AWS
accounts often cannot reserve concurrency; the script carries on without it.

---

## Optional: a custom domain

ACM certificates for CloudFront **must** live in `us-east-1`, whatever region
the bucket is in.

```bash
export DOMAIN=quiz.example.edu

export CERT_ARN=$(aws acm request-certificate \
  --region us-east-1 \
  --domain-name "$DOMAIN" \
  --validation-method DNS \
  --query CertificateArn --output text)

# Add the CNAME it asks for to your DNS, then wait:
aws acm describe-certificate --region us-east-1 --certificate-arn "$CERT_ARN" \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord'
aws acm wait certificate-validated --region us-east-1 --certificate-arn "$CERT_ARN"
```

Then attach it to the distribution (console: **CloudFront → your distribution →
Settings → Edit →** add the Alternate domain name and select the certificate),
and point a DNS `CNAME` (or a Route 53 `A`-record alias) at `$DIST_DOMAIN`.
Finally add the custom domain to Firebase's authorized domains too.

---

## Costs

A class of 300 students is comfortably inside the free tiers.

| Service | Usage | Cost |
|---|---|---|
| S3 | ~1 MB stored, a few thousand GETs | pennies |
| CloudFront | 1 TB/month + 10M requests free | $0 |
| Firestore | 50k reads / 20k writes per day free | $0 |
| Lambda (guidance) | 512 MB × ~60 s per attempt; 400k GB-s/month free ≈ 13,000 attempts | $0 |
| Gemini API | one call per attempt | free tier, within its rate limits |

One attempt = ~16 document reads (the bank) + 1 aggregation read (the prior-
attempt count) + 1 write. 300 students ≈ 5,100 reads. Set
`VITE_USE_FIRESTORE_QUESTIONS=false` to serve the bundled bank instead and drop
that to ~1 read per attempt.

Google sign-in is free at any volume.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AccessDenied` XML at the root | Bucket policy not applied, or `DIST_ID` wrong in it | Redo step 6c |
| Blank page, console 404 on `/assets/...` | `--delete` sync removed assets a cached `index.html` still names | Re-run step 5, then invalidate |
| `auth/unauthorized-domain` on the deployed site | CloudFront domain not in Firebase's authorized list | Step 7 |
| Sign-in button does nothing | Popup blocked; the app falls back to redirect, which needs step 7 done | Allow popups, or complete step 7 |
| Orange "could not be saved" banner | Google provider off, domain not authorised, or rules not deployed | Steps 1.3, 7, 2 |
| Old version after deploy | Invalidation skipped | `aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*"` |
| `Missing or insufficient permissions` in console | Rules deployed but anonymous auth disabled | Step 1.3 |
| Deep link 404s on refresh | `CustomErrorResponses` missing | Step 6b |

---

## Teardown

```bash
aws cloudfront get-distribution-config --id "$DIST_ID" > /tmp/d.json
# set "Enabled": false in /tmp/d.json, then:
aws cloudfront update-distribution --id "$DIST_ID" \
  --distribution-config "$(jq '.DistributionConfig' /tmp/d.json)" \
  --if-match "$(jq -r '.ETag' /tmp/d.json)"
aws cloudfront wait distribution-deployed --id "$DIST_ID"
aws cloudfront delete-distribution --id "$DIST_ID" --if-match "<new ETag>"

aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET"
```
