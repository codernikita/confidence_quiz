/**
 * Firebase ID-token check for the deployed guidance API.
 *
 * The Lambda function URL is public, and every call spends Gemini quota and
 * Lambda time. Requiring the token that Firebase sign-in already gives each
 * student limits it to people signed in to this quiz's Firebase project.
 *
 * Verified locally against Google's published signing keys (fetched once and
 * cached by jose), per Firebase's documented rules for ID tokens: RS256, issuer
 * https://securetoken.google.com/<project>, audience <project>, a subject, and
 * not expired.
 */
import { createRemoteJWKSet, jwtVerify } from "jose";

const JWKS = createRemoteJWKSet(
  new URL(
    "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com",
  ),
);

export class AuthError extends Error {
  status = 401;
}

/** Returns the verified token payload, or throws AuthError. */
export async function verifyFirebaseToken(authorizationHeader, projectId) {
  const match = /^Bearer\s+(.+)$/i.exec(String(authorizationHeader || ""));
  if (!match) throw new AuthError("Sign in to get study guidance.");
  try {
    const { payload } = await jwtVerify(match[1], JWKS, {
      issuer: `https://securetoken.google.com/${projectId}`,
      audience: projectId,
      algorithms: ["RS256"],
    });
    if (!payload.sub) throw new Error("token has no subject");
    return payload;
  } catch {
    throw new AuthError("Your sign-in has expired. Reload the page and try again.");
  }
}
