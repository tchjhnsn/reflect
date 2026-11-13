import type { Metadata } from "next";
import Link from "next/link";
import { LISTS } from "@reflect/core/lists";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { LearnHeader } from "../header";

export const metadata: Metadata = { title: "Lists" };

/** Lists of books chosen by others, each with who chose it and when. */
export default async function ListsPage() {
  await requireUser();
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <section className="mt-8 space-y-4">
        <h2 className="text-lg font-semibold">Lists</h2>
        <ul className="space-y-3">
          {LISTS.map((l) => (
            <li key={l.id} className="rounded-md border p-4">
              <Link href={`/learn/lists/${l.id}`} className="font-medium hover:underline">{l.name}</Link>
              <p className="text-muted-foreground mt-1 text-sm">{l.about}</p>
              <p className="text-muted-foreground mt-1 text-xs">{l.books.length} books · {l.source}</p>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
