import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { publishedQuote } from "@/server/published-store";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const q = await publishedQuote((await params).id).catch(() => null);
  if (!q) return { title: "Quote" };
  const by = [q.author, q.title].filter(Boolean).join(", ");
  return { title: by, description: q.quote.slice(0, 200), openGraph: { title: by, description: q.quote.slice(0, 200) } };
}

/**
 * A quote the person chose to publish, as a card with its citation. Public: anyone with the link sees it, and only
 * it. Taken down, the page is gone.
 */
export default async function QuotePage({ params }: Props) {
  const q = await publishedQuote((await params).id).catch(() => null);
  if (!q) notFound();
  return (
    <main className="flex min-h-svh items-center justify-center px-4 py-16">
      <figure className="bg-card w-full max-w-2xl rounded-xl border p-8 shadow-sm sm:p-12">
        <blockquote className="font-serif text-2xl leading-relaxed text-pretty sm:text-3xl">“{q.quote}”</blockquote>
        <figcaption className="text-muted-foreground mt-6 text-sm">
          {q.author ? <span className="text-foreground">{q.author}</span> : null}
          {q.author ? ", " : ""}
          {q.work ? <Link href={`/learn/works/${q.work}`} className="italic underline-offset-4 hover:underline">{q.title}</Link> : <span className="italic">{q.title}</span>}
          {q.place ? `, ${q.place}` : ""}
        </figcaption>
      </figure>
    </main>
  );
}
