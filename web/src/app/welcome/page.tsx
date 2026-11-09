import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { hasOnboarded } from "@/server/person-store";
import { Onboarding } from "./onboarding";
import { keepAnswers } from "./actions";

export const metadata: Metadata = { title: "Welcome to Reflect" };
export const dynamic = "force-dynamic";

export default async function WelcomePage() {
  const user = await requireUser();
  if (await hasOnboarded(user.id)) redirect("/journal");
  return <Onboarding keep={keepAnswers} />;
}
