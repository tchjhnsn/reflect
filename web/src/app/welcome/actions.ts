"use server";

import { redirect } from "next/navigation";
import { ONBOARDING } from "@/core/structure";
import { EMPTY_PERSON, type IsoDate } from "@/core/person";
import { START, finish, reduce, type OnboardingState } from "@/core/onboarding";
import { requireUser } from "@/server/session";
import { saveOnboarding } from "@/server/person-store";

function today(): IsoDate {
  return new Date().toISOString().slice(0, 10) as IsoDate;
}

/** Re-run the answers through the core on the server, so only what the core accepts is written. */
export async function keepAnswers(answers: Record<string, string>): Promise<{ error: string } | never> {
  const user = await requireUser();
  const at = today();
  let state: OnboardingState = START;
  for (const q of ONBOARDING) {
    const text = typeof answers[q.id] === "string" ? answers[q.id]! : "";
    state = reduce(state, { type: "answer", text, today: at });
    if (state.error) return { error: state.error.message };
  }
  await saveOnboarding(user.id, finish(state, EMPTY_PERSON, at), at);
  redirect("/journal");
}
