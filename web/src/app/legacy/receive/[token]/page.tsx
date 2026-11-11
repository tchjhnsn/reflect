import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { receive } from "@/server/legacy-store";

export const metadata: Metadata = { title: "For you", robots: { index: false } };
export const dynamic = "force-dynamic";

/** A recipient's part, once released. Before that the link shows nothing, not even that something waits. */
export default async function ReceivePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  const r = await receive(token);
  if (!r) notFound();
  return (
    <main className="shell">
      <section className="screen">
        <p className="eyebrow">From {r.from}</p>
        {r.items.map((item, i) => (
          <article key={i}>
            <h1 className="prompt">{item.title}</h1>
            {item.text.split(/\n{2,}/).map((p, j) => (
              <p key={j} className="note" style={{ whiteSpace: "pre-wrap" }}>{p}</p>
            ))}
          </article>
        ))}
      </section>
    </main>
  );
}
