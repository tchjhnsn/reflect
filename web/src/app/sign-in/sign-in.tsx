"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

/** Where to go next when this sign-in was for a person's AI app, else null. */
function oauthNext(data: unknown): string | null {
  if (typeof data !== "object" || data === null || !("redirect" in data) || !("url" in data)) return null;
  return typeof data.url === "string" ? data.url : null;
}

export function SignIn() {
  const router = useRouter();
  const [mode, setMode] = useState<"sign-up" | "sign-in">("sign-up");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setPending(true);
    setError(null);
    const result =
      mode === "sign-up"
        ? await authClient.signUp.email({ email, password, name: "" })
        : await authClient.signIn.email({ email, password });
    setPending(false);
    if (result.error) setError(result.error.message ?? "That didn't work. Try again.");
    // Signing in for a person's AI app: the server sends the browser on to /consent or back to the app.
    else {
      const next = oauthNext(result.data);
      if (next) window.location.href = next;
      else {
        // Sent here from a page that needs an account (an invitation to examine someone): back to it after.
        const back = new URLSearchParams(window.location.search).get("next");
        router.replace(back && back.startsWith("/") && !back.startsWith("//") ? back : "/");
      }
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardDescription>Reflect</CardDescription>
          <CardTitle className="text-xl">{mode === "sign-up" ? "Create your account" : "Welcome back"}</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={submit}>
            <div className="space-y-1.5">
              <label className="text-sm font-medium" htmlFor="email">Email</label>
              <Input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium" htmlFor="password">Password</label>
              <Input
                id="password"
                type="password"
                autoComplete={mode === "sign-up" ? "new-password" : "current-password"}
                minLength={10}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error ? <p className="text-destructive text-sm" role="alert">{error}</p> : null}
            <Button type="submit" className="w-full" disabled={pending}>{mode === "sign-up" ? "Create account" : "Sign in"}</Button>
            <Button type="button" variant="ghost" className="w-full" onClick={() => setMode(mode === "sign-up" ? "sign-in" : "sign-up")}>
              {mode === "sign-up" ? "I have an account" : "Create an account"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  );
}
