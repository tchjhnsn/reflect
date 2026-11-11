import Link from "next/link";

/** The header on app pages: an eyebrow on the left; the journal, what you know, your ideas, and your initial on the right. */
export function Nav({ current, eyebrow, initial }: { current: "journal" | "learn" | "ideas" | "me"; eyebrow: string; initial: string }) {
  return (
    <nav className="top" aria-label="Main">
      <span className="eyebrow">{eyebrow}</span>
      <div className="top-links">
        <Link href="/journal" aria-current={current === "journal" ? "page" : undefined}>
          Journal
        </Link>
        <Link href="/learn" aria-current={current === "learn" ? "page" : undefined}>
          Learn
        </Link>
        <Link href="/ideas" aria-current={current === "ideas" ? "page" : undefined}>
          Ideas
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
