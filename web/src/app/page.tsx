import type { Metadata } from "next";
import Link from "next/link";
import { mono } from "./fonts";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { GLYPHS, hrefOf, MINE, OURS, toneOf, type Facts, type Tone } from "./home-tiles";

export const metadata: Metadata = { title: "Reflect" };
export const dynamic = "force-dynamic";

const FILL: Record<Tone, string> = { action: "var(--home-action)", thought: "var(--home-thought)", unstated: "var(--home-unstated)" };
const SAY: Record<Tone, string> = { action: "in action", thought: "in thought", unstated: "unstated" };

/** Home: your life as objects. MINE is the private vault; OURS, the shared spaces, is not built yet. */
export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const ours = (await searchParams).view === "ours";
  const learn = await learnView(user.id);
  const facts: Facts = { learn: { underway: learn.books.some((b) => b.status === "reading"), stated: learn.books.length > 0 || learn.known.length > 0 } };
  const tiles = ours ? OURS : MINE;
  return (
    <main className={`home ${mono.className}`}>
      <div className="home-bar" aria-hidden="true" />
      <header className="home-top">
        <Link href="/chat" className="home-icon" aria-label="Ask Reflect">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M11 2v18M2 11h18" stroke="currentColor" strokeWidth="2" fill="none" /></svg>
        </Link>
        <nav className="home-toggle" aria-label="Whose">
          <Link href="/" aria-current={!ours ? "page" : undefined}>MINE</Link>
          <Link href="/?view=ours" aria-current={ours ? "page" : undefined}>OURS</Link>
        </nav>
        <span className="home-icon is-off" aria-label="Your journal, not built yet in this design" title="Journal: not built yet">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M3 7h6l2 2h8v10H3z" stroke="currentColor" strokeWidth="2" fill="none" strokeLinejoin="round" /></svg>
        </span>
      </header>
      {ours ? <p className="home-note">Shared spaces (a marriage, a household, a village) are not built yet.</p> : null}
      <ul className="home-grid">
        {tiles.map((t) => {
          const tone = ours ? "unstated" : toneOf(t, facts);
          const href = ours ? null : hrefOf(t);
          const body = (
            <>
              <svg viewBox="0 0 100 100" aria-hidden="true"><path d={GLYPHS[t.id]} fill={FILL[tone]} fillRule="evenodd" /></svg>
              <span className="home-label">{t.label.toUpperCase()}</span>
            </>
          );
          return (
            <li key={t.id}>
              {href ? (
                <Link href={href} className="home-tile" aria-label={`${t.label}, ${SAY[tone]}`}>{body}</Link>
              ) : (
                <span className="home-tile is-off" aria-label={`${t.label}, ${SAY[tone]}, not built yet`}>{body}</span>
              )}
            </li>
          );
        })}
      </ul>
      <footer className="home-foot">
        <span><i style={{ background: FILL.unstated }} />UNSTATED</span>
        <span><i style={{ background: FILL.thought }} />IN THOUGHT</span>
        <span><i style={{ background: FILL.action }} />IN ACTION</span>
      </footer>
    </main>
  );
}
