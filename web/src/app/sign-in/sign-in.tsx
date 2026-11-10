"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";

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
    else router.replace("/journal");
  };

  return (
    <main className="shell">
      <form className="screen" onSubmit={submit}>
        <p className="eyebrow">Reflect</p>
        <h1 className="prompt">{mode === "sign-up" ? "Start your journal." : "Welcome back."}</h1>
        <label className="note" htmlFor="email">
          Email
        </label>
        <input id="email" className="field" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <label className="note" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          className="field"
          type="password"
          autoComplete={mode === "sign-up" ? "new-password" : "current-password"}
          minLength={10}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="actions">
          <button type="button" className="quiet" onClick={() => setMode(mode === "sign-up" ? "sign-in" : "sign-up")}>
            {mode === "sign-up" ? "I have an account" : "Create an account"}
          </button>
          <button type="submit" className="primary" disabled={pending}>
            {mode === "sign-up" ? "Create account" : "Sign in"}
          </button>
        </div>
      </form>
    </main>
  );
}
