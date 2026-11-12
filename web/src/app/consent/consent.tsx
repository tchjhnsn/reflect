"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

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
    <main className="flex min-h-screen items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardDescription>Reflect</CardDescription>
          <CardTitle className="text-xl">Connect an app to Reflect?</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p>{name ? `An app that calls itself “${name}”` : "An app"} is asking to use Reflect for you. It will be able to:</p>
          <ul className="list-disc space-y-1 pl-5">
            {asks.map((s) => (
              <li key={s}>{SAYS[s]}</li>
            ))}
          </ul>
          <p className="text-muted-foreground">Nothing it suggests is kept unless you say yes to it. You can disconnect it at any time.</p>
          {error ? <p className="text-destructive" role="alert">{error}</p> : null}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" disabled={pending} onClick={() => answer(false)}>Don’t allow</Button>
            <Button type="button" disabled={pending} onClick={() => answer(true)}>Allow</Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
