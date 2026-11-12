/**
 * Open mode: no sign-in, everything belongs to one owner account. On only where nobody else can reach the
 * app: the local dev server, and Vercel previews (behind the Vercel team's own sign-in). Production keeps
 * sign-in. REFLECT_OPEN=off turns it off anywhere; REFLECT_OPEN=on turns it on (for a local `pnpm start`).
 */
export function openMode(env: Record<string, string | undefined> = process.env): boolean {
  if (env.REFLECT_OPEN === "off") return false;
  if (env.REFLECT_OPEN === "on") return true;
  return env.VERCEL_ENV === "preview" || (env.NODE_ENV === "development" && !env.VERCEL);
}
