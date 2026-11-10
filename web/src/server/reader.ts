import { getVercelOidcToken } from "@vercel/oidc";
import { jevEndpoint, readSalience as read } from "@reflect/core/reader";
import type { Salience } from "@reflect/core/state";

export { JEV_GUESS_AT } from "@reflect/core/reader";

/** The core reader, with the deployment's OIDC token for AI Gateway when it runs on Vercel. */
export async function readSalience(text: string): Promise<Salience | null> {
  const oidc = process.env.VERCEL && !process.env.AI_GATEWAY_API_KEY ? await getVercelOidcToken().catch(() => undefined) : undefined;
  return read(text, jevEndpoint(oidc));
}
