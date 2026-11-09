import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { auth } from "./auth";

export async function requireUser(): Promise<{ id: string }> {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/sign-in");
  return { id: session.user.id };
}
