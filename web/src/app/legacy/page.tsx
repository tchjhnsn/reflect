import type { Metadata } from "next";
import { requireUser } from "@/server/session";
import { legacyStatus } from "@/server/legacy-store";
import { siteUrl } from "@/server/site";
import { stopMine } from "./actions";
import { releaseText } from "./release-text";

export const metadata: Metadata = { title: "Legacy" };
export const dynamic = "force-dynamic";

export default async function LegacyPage() {
  const user = await requireUser();
  const s = await legacyStatus(user.id);
  const running = s.release.state === "confirming" || s.release.state === "waiting";
  return (
    <main className="shell">
      <section className="screen">
        <p className="eyebrow">Legacy</p>
        <h1 className="prompt">What goes to whom after you die</h1>
        {s.plan ? (
          <>
            <p className="note" role="status">{releaseText(s.release)}</p>
            {running ? (
              <form action={stopMine}>
                <p className="note">If you are reading this, you can stop it.</p>
                <button type="submit" className="primary">Stop the release</button>
              </form>
            ) : null}
            <p className="note">
              {s.plan.confirmations} of your {s.trustees.length} trustees must confirm your death, then {s.plan.waitDays} days pass in which you or any trustee can stop it. Only then does each person receive their part.
            </p>
            <h2 className="lbl">Your trustees</h2>
            <p className="note">Give each trustee their own link. Anyone with it can confirm or stop a release, so give it only to them.</p>
            <ul className="note">
              {s.trustees.map((t) => (
                <li key={t.token}>
                  {t.name} ({t.contact}): <code>{`${siteUrl()}/legacy/trustee/${t.token}`}</code>
                </li>
              ))}
            </ul>
            <h2 className="lbl">Who receives something</h2>
            <ul className="note">
              {s.gifts.map((g, i) => (
                <li key={i}>
                  {g.to} ({g.contact})
                </li>
              ))}
            </ul>
            <p className="note muted">Recipients get their links from your trustees once the release happens. Before then their part is sealed and nobody sees it.</p>
          </>
        ) : (
          <p className="note">
            You have no release plan here yet. Write it in your vault with <code>reflect release</code>, then ask your AI to send it to Reflect; or ask your AI, connected to Reflect, to keep one for you. Only what you made to survive you is ever sent.
          </p>
        )}
      </section>
    </main>
  );
}
