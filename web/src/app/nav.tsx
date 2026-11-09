import Link from "next/link";

/** The header on app pages: an eyebrow on the left, the journal and your initial on the right. */
export function Nav({ current, eyebrow, initial }: { current: "journal" | "me"; eyebrow: string; initial: string }) {
  return (
    <nav className="top" aria-label="Main">
      <span className="eyebrow">{eyebrow}</span>
      <div className="top-links">
        <Link href="/journal" aria-current={current === "journal" ? "page" : undefined}>
          Journal
        </Link>
        <Link href="/me" className="avatar" aria-label="Me" aria-current={current === "me" ? "page" : undefined}>
          {initial}
        </Link>
      </div>
    </nav>
  );
}

export function initialOf(name: string | null): string {
  return (name?.trim()[0] ?? "·").toUpperCase();
}
