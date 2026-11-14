"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { addEvent, removeEvent, saveEvent } from "@/server/calendar-store";

const back = (form: FormData) => {
  const b = String(form.get("$back") ?? "");
  return b.startsWith("/calendar") ? b : "/calendar";
};
const fields = (form: FormData) => ({ title: form.get("title"), tile: form.get("tile"), day: form.get("day"), starts: form.get("starts"), ends: form.get("ends"), repeat: form.get("repeat"), until: form.get("until"), note: form.get("note"), recordId: form.get("recordId") });

export async function addEventAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await addEvent(user.id, fields(form));
  revalidatePath("/calendar");
  redirect(back(form));
}

export async function saveEventAction(id: string, form: FormData): Promise<void> {
  const user = await requireUser();
  await saveEvent(user.id, id, fields(form));
  revalidatePath("/calendar");
  redirect(back(form));
}

export async function removeEventAction(id: string, form: FormData): Promise<void> {
  const user = await requireUser();
  await removeEvent(user.id, id, true);
  revalidatePath("/calendar");
  const b = back(form);
  redirect(`${b}${b.includes("?") ? "&" : "?"}removed=${id}`);
}

export async function restoreEventAction(id: string, form: FormData): Promise<void> {
  const user = await requireUser();
  await removeEvent(user.id, id, false);
  revalidatePath("/calendar");
  redirect(back(form).replace(/[?&]removed=[^&]+/, ""));
}
