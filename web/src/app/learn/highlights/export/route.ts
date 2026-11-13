import { excerptsMarkdown } from "@reflect/core/excerpts";
import { requireUser } from "@/server/session";
import { allExcerpts } from "@/server/excerpt-store";
import { highlightBooks } from "@/server/highlights";

/** All the person's highlights and excerpts as one Markdown file to download, by book, with places and notes. */
export async function GET(): Promise<Response> {
  const user = await requireUser();
  const md = excerptsMarkdown(await highlightBooks(user.id, await allExcerpts(user.id)));
  return new Response(md, { headers: { "content-type": "text/markdown; charset=utf-8", "content-disposition": 'attachment; filename="highlights.md"', "cache-control": "no-store" } });
}
