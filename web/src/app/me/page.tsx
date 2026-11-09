import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { hasOnboarded } from "@/server/person-store";
import { loadMe } from "@/server/me-store";
import { MePage } from "./me";

export const metadata: Metadata = { title: "Me" };
export const dynamic = "force-dynamic";

export default async function Page() {
  const user = await requireUser();
  if (!(await hasOnboarded(user.id))) redirect("/welcome");
  return <MePage me={await loadMe(user.id)} />;
}
