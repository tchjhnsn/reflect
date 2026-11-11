"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { isStance, type Target } from "@reflect/core/ideas";
import { requireUser } from "@/server/session";
import { getIdea, keepIdea, resolveByTitle } from "@/server/ideas-store";

const opt = (form: FormData, k: string) => String(form.get(k) ?? "").trim();

/** What the person picked from what they kept, or else named in their own words (found by title when it matches). */
async function picked(userId: string, form: FormData): Promise<Target> {
  const ref = opt(form, "ref");
  const url = opt(form, "url");
  const t: Target = ref ? { ref } : await resolveByTitle(userId, opt(form, "text"));
  return url && !t.url ? { ...t, url } : t;
}

async function titleOf(userId: string, form: FormData): Promise<string> {
  const idea = await getIdea(userId, opt(form, "idea"));
  if (!idea) throw new Error("no such idea");
  return idea.title;
}

export async function newIdeaAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const from = await Promise.all(String(form.get("from") ?? "").split("\n").map((x) => x.trim()).filter(Boolean).map(async (x) => ({ target: await resolveByTitle(user.id, x) })));
  const id = await keepIdea(user.id, opt(form, "title"), { statement: opt(form, "statement"), from });
  revalidatePath("/ideas");
  redirect(`/ideas/${id}`);
}

export async function restateAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await keepIdea(user.id, await titleOf(user.id, form), { statement: opt(form, "statement") });
  revalidatePath(`/ideas/${opt(form, "idea")}`);
}

export async function addInputAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const x = { target: await picked(user.id, form), ...(opt(form, "how") ? { how: opt(form, "how") } : {}) };
  await keepIdea(user.id, await titleOf(user.id, form), opt(form, "list") === "maybe" ? { maybe: [x] } : { from: [x] });
  revalidatePath(`/ideas/${opt(form, "idea")}`);
}

export async function addRecommendationAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const stance = opt(form, "stance");
  if (!isStance(stance)) throw new Error("choose where it stands");
  await keepIdea(user.id, await titleOf(user.id, form), { related: [{ target: await picked(user.id, form), stance, by: opt(form, "by") || "me", ...(opt(form, "note") ? { note: opt(form, "note") } : {}) }] });
  revalidatePath(`/ideas/${opt(form, "idea")}`);
}

export async function dropAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const t = JSON.parse(opt(form, "target")) as Target;
  await keepIdea(user.id, await titleOf(user.id, form), { drop: [t.ref ? { ref: t.ref } : { text: t.text ?? "" }] });
  revalidatePath(`/ideas/${opt(form, "idea")}`);
}
