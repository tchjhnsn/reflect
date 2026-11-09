"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";

export function Nav({ current }: { current: "journal" | "me" }) {
  const router = useRouter();
  return (
    <nav className="nav" aria-label="Main">
      <span className="brand">Reflect</span>
      <div className="nav-links">
        <Link href="/journal" aria-current={current === "journal" ? "page" : undefined}>
          Journal
        </Link>
        <Link href="/me" aria-current={current === "me" ? "page" : undefined}>
          Me
        </Link>
        <button
          type="button"
          className="quiet"
          onClick={async () => {
            await authClient.signOut();
            router.replace("/sign-in");
          }}
        >
          Sign out
        </button>
      </div>
    </nav>
  );
}
