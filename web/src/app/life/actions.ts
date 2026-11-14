"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { kindOf } from "@reflect/core/life";
import { requireUser } from "@/server/session";
import { addLifeRecord, keepLifeAct, lifeRecord, removeLifeRecord, restoreLifeRecord, saveLifeRecord, setLifeState } from "@/server/life-store";

const fields = (form: FormData) => Object.fromEntries([...form.entries()].filter(([k]) => !k.startsWith("$")).map(([k, v]) => [k, String(v)]));
const back = (form: FormData, fallback: string) => {
  const to = String(form.get("$back") ?? "");
  return to.startsWith("/life/") ? to : fallback;
};

/** Keeps a new record; a kind kept once stays on its app's page, any other opens its own page. */
export async function addLifeAction(app: string, kind: string, form: FormData): Promise<void> {
  const user = await requireUser();
  const id = await addLifeRecord(user.id, app, kind, fields(form), String(form.get("$state") ?? "") || undefined);
  revalidatePath("/", "layout");
  if (kindOf(app as never, kind)?.one) redirect(back(form, `/life/${app}?k=${kind}`));
  redirect(`/life/${app}/${id}`);
}

export async function saveLifeAction(id: string, form: FormData): Promise<void> {
  const user = await requireUser();
  await saveLifeRecord(user.id, id, fields(form));
  revalidatePath("/", "layout");
  redirect(back(form, `/life`));
}

export async function stateLifeAction(id: string, state: string, to: string): Promise<void> {
  const user = await requireUser();
  await setLifeState(user.id, id, state);
  revalidatePath("/", "layout");
  redirect(to.startsWith("/life/") ? to : "/");
}

export async function actLifeAction(id: string, form: FormData): Promise<void> {
  const user = await requireUser();
  await keepLifeAct(user.id, id, { kind: String(form.get("kind") ?? ""), day: String(form.get("day") ?? ""), minutes: form.get("minutes") ?? undefined, amount: form.get("amount") ?? undefined, note: form.get("note") ?? undefined });
  revalidatePath("/", "layout");
  redirect(back(form, "/"));
}

/** Removes a record and returns to its app, where Undo restores it. */
export async function removeLifeAction(id: string): Promise<void> {
  const user = await requireUser();
  const r = await lifeRecord(user.id, id);
  if (!r) redirect("/");
  await removeLifeRecord(user.id, id);
  revalidatePath("/", "layout");
  redirect(`/life/${r.app}?k=${r.kind}&removed=${id}`);
}

export async function restoreLifeAction(id: string, to: string): Promise<void> {
  const user = await requireUser();
  await restoreLifeRecord(user.id, id);
  revalidatePath("/", "layout");
  redirect(to.startsWith("/life/") ? to : "/");
}
