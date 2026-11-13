import { HARDCOVER_URL, hardcoverAuth, hardcoverBook, type HardcoverBook } from "@reflect/core/hardcover";

type Answer = { data?: unknown; errors?: { message?: string }[]; error?: string; error_description?: string };

/**
 * A work's editions ranked by Hardcover readers, when `HARDCOVER_TOKEN` is set. Answers are kept a day, since the
 * free API allows 5,000 requests a day. `error` says why there are none when a query failed or no token is set:
 * Hardcover's own message, or the start of a page that is not its API's answer (never the token).
 */
export async function hardcoverFor(work: { title: string; authors: readonly string[]; isbns: readonly string[] }): Promise<{ book: HardcoverBook | null; error?: string }> {
  const token = process.env.HARDCOVER_TOKEN;
  if (!token?.trim()) return { book: null, error: "HARDCOVER_TOKEN is not set in this deployment" };
  let error: string | undefined;
  const query = async (q: string, variables: Record<string, unknown>) => {
    try {
      const r = await fetch(HARDCOVER_URL, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json", authorization: hardcoverAuth(token), "user-agent": "reflect-app (book pages: edition rankings)" },
        body: JSON.stringify({ query: q, variables }),
        next: { revalidate: 86400 },
      });
      const text = await r.text();
      let body: Answer | null = null;
      try {
        body = JSON.parse(text) as Answer;
      } catch {
        body = null;
      }
      if (!r.ok || !body || body.errors?.length || body.error) {
        error = `HTTP ${r.status}: ${body?.errors?.map((e) => e.message).join("; ") || body?.error_description || body?.error || text.replace(/\s+/g, " ").slice(0, 160)}`.slice(0, 300);
        return null;
      }
      return body.data ?? null;
    } catch (e) {
      error = `could not reach Hardcover: ${e instanceof Error ? e.message : String(e)}`.slice(0, 300);
      return null;
    }
  };
  const book = await hardcoverBook(work, query).catch(() => null);
  return { book, ...(error ? { error } : !book ? { error: "Hardcover does not have this book" } : {}) };
}
