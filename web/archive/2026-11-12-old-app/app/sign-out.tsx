"use client";

import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";

export function SignOut() {
  const router = useRouter();
  return (
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
  );
}
