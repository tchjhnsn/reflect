import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { db } from "@/db/pool";
import { auth } from "./auth";
import { openMode } from "./open-mode";

export const OWNER_ID = "reflect-owner";
let ownerReady: Promise<void> | null = null;
const ensureOwner = () =>
  (ownerReady ??= db()
    .query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, 'Owner', 'owner@reflect.local', false, now(), now()) ON CONFLICT (id) DO NOTHING`, [OWNER_ID])
    .then(() => undefined)
    .catch((e: unknown) => {
      ownerReady = null;
      throw e;
    }));

export async function requireUser(): Promise<{ id: string }> {
  if (openMode()) {
    await ensureOwner();
    return { id: OWNER_ID };
  }
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/sign-in");
  return { id: session.user.id };
}
