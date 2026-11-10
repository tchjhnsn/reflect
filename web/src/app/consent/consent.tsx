"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";

/** What each scope lets the app do, in the person's words. */
const SAYS: Record<string, string> = {
  journal: "Read your journal entries, write new ones, and keep the readings you approve.",
  profile: "See your name.",
  offline_access: "Stay connected until you disconnect it.",
};

export function Consent({ name, scope }: { name: string | null; scope: string }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const asks = Object.keys(SAYS).filter((s) => scope.split(" ").includes(s));

  const answer = async (accept: boolean) => {
    setPending(true);
    setError(null);
    const result = await authClient.oauth2.consent({ accept });
    setPending(false);
    const url = (result.data as { url?: string } | null)?.url;
    if (result.error) setError(result.error.message ?? "That didn't work. Try again.");
    else if (url) window.location.href = url;
  };

  return (
    <main className="shell">
      <section className="screen">
        <p className="eyebrow">Reflect</p>
        <h1 className="prompt">Connect an app to your journal?</h1>
        <p className="note">{name ? `An app that calls itself “${name}”` : "An app"} is asking to use Reflect for you. It will be able to:</p>
        <ul className="note">
          {asks.map((s) => (
            <li key={s}>{SAYS[s]}</li>
          ))}
        </ul>
        <p className="note">Nothing it suggests is kept unless you say yes to it. You can disconnect it at any time.</p>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="actions">
          <button type="button" className="quiet" disabled={pending} onClick={() => answer(false)}>
            Don’t allow
          </button>
          <button type="button" className="primary" disabled={pending} onClick={() => answer(true)}>
            Allow
          </button>
        </div>
      </section>
    </main>
  );
}
