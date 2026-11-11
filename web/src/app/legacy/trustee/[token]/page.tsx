import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { trusteeView } from "@/server/legacy-store";
import { siteUrl } from "@/server/site";
import { confirmAsTrustee, stopAsTrustee } from "../../actions";
import { releaseText } from "../../release-text";

export const metadata: Metadata = { title: "Trustee", robots: { index: false } };
export const dynamic = "force-dynamic";

export default async function TrusteePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  const v = await trusteeView(token);
  if (!v) notFound();
  const running = v.release.state === "confirming" || v.release.state === "waiting";
  return (
    <main className="shell">
      <section className="screen">
        <p className="eyebrow">Reflect</p>
        <h1 className="prompt">{v.person} named you, {v.trustee}, a trustee</h1>
        <p className="note">
          When {v.person} dies, their trustees confirm it here. Once enough have, a waiting period begins in which {v.person} or any trustee can stop it. Then you will see who receives what, with a link to give each of them.
        </p>
        <p className="note" role="status">{releaseText(v.release)}</p>
        {v.release.state !== "released" && !v.confirmedByYou ? (
          <form action={confirmAsTrustee}>
            <input type="hidden" name="token" value={token} />
            <label className="note">
              <input type="checkbox" name="sure" required /> I confirm that {v.person} has died.
            </label>
            <div className="actions">
              <button type="submit" className="primary">Confirm</button>
            </div>
          </form>
        ) : null}
        {v.confirmedByYou && v.release.state !== "released" ? <p className="note">You have confirmed.</p> : null}
        {running ? (
          <form action={stopAsTrustee}>
            <input type="hidden" name="token" value={token} />
            <p className="note">If this is a mistake, stop it.</p>
            <button type="submit" className="quiet">Stop the release</button>
          </form>
        ) : null}
        {v.recipients.length ? (
          <>
            <h2 className="lbl">Give each person their link</h2>
            <ul className="note">
              {v.recipients.map((r) => (
                <li key={r.token}>
                  {r.to} ({r.contact}): <code>{`${siteUrl()}/legacy/receive/${r.token}`}</code>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </section>
    </main>
  );
}
