import Link from "next/link";
import { BELIEF_KINDS, type BookEntry } from "@reflect/core/book";
import { originLabel } from "@reflect/core/origins";

const month = (iso: string) => new Date(iso.length === 10 ? `${iso}T12:00:00` : iso).toLocaleDateString(undefined, { month: "long", year: "numeric" });

/** One belief as the book and the domain pages show it: its words, and what it serves, comes from and rests on. */
export function BeliefLine({ e }: { e: BookEntry }) {
  const meta = [
    e.state === "held" ? `Held since ${month(e.since)}` : `Once held, ${month(e.since)} to ${month(e.until!)}`,
    e.origins.length ? `from ${e.origins.map(originLabel).join(", ").toLowerCase()}` : null,
    e.holders.length ? `with ${e.holders.map((h) => (h.citation ? `${h.name}, ${h.citation}` : h.name)).join("; ")}` : null,
  ].filter(Boolean);
  return (
    <div className="space-y-1">
      <p className={e.state === "once" ? "text-muted-foreground font-serif text-lg leading-relaxed line-through decoration-1" : "font-serif text-lg leading-relaxed"}>{e.text}</p>
      {e.kind === "practical" && e.state === "held" ? (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 text-sm">
          <dt className="text-muted-foreground">Serves</dt><dd className="font-serif italic">{e.serves?.text}</dd>
          {e.circumstances ? <><dt className="text-muted-foreground">For</dt><dd>{e.circumstances}</dd></> : null}
        </dl>
      ) : null}
      <p className="text-muted-foreground text-xs">
        {meta.join(" · ")}
        {e.source ? <> · in <Link href={`/journal/w/${e.source.workId}`} className="underline underline-offset-2">{e.source.title}</Link></> : null}
      </p>
    </div>
  );
}

/** A chapter's beliefs in groups: ideals, practical beliefs, not yet marked, and once held. */
export function groups(entries: readonly BookEntry[]) {
  const held = entries.filter((e) => e.state === "held");
  return [
    ...BELIEF_KINDS.map((k) => ({ id: k.id, title: k.plural, about: k.about, entries: held.filter((e) => e.kind === k.id) })),
    { id: "unmarked", title: "Not yet marked", about: "Beliefs you hold that you have not called an ideal or a practical belief.", entries: held.filter((e) => !e.kind) },
    { id: "once", title: "Once held", about: "How your mind changed.", entries: entries.filter((e) => e.state === "once") },
  ].filter((g) => g.entries.length);
}
