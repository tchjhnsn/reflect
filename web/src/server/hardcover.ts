import { HARDCOVER_URL, hardcoverAuth, hardcoverBook, type HardcoverBook } from "@reflect/core/hardcover";

/**
 * A work's editions ranked by Hardcover readers, when `HARDCOVER_TOKEN` is set. Answers are kept a day, since the
 * free API allows 5,000 requests a day. `error` is Hardcover's own message when a query fails (never the token).
 */
export async function hardcoverFor(work: { title: string; authors: readonly string[]; isbns: readonly string[] }): Promise<{ book: HardcoverBook | null; error?: string }> {
  const token = process.env.HARDCOVER_TOKEN;
  if (!token?.trim()) return { book: null };
  let error: string | undefined;
  const query = async (q: string, variables: Record<string, unknown>) => {
    const r = await fetch(HARDCOVER_URL, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: hardcoverAuth(token), "user-agent": "reflect-app (book pages: edition rankings)" },
      body: JSON.stringify({ query: q, variables }),
      next: { revalidate: 86400 },
    });
    const body = (await r.json().catch(() => null)) as { data?: unknown; errors?: { message?: string }[]; error?: string; error_description?: string } | null;
    if (!r.ok || body?.errors?.length || body?.error) {
      error = (body?.errors?.map((e) => e.message).join("; ") || body?.error_description || body?.error || `HTTP ${r.status}`).slice(0, 300);
      return null;
    }
    return body?.data ?? null;
  };
  const book = await hardcoverBook(work, query).catch(() => null);
  return { book, ...(error ? { error } : {}) };
}
