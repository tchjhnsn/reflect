"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { requireUser } from "@/server/session";
import { createWork, saveWork } from "@/server/work-store";

/** Start a work from a template of the prompt library and open it in the editor. */
export async function createWorkAction(templateId: string, about: string | null): Promise<void> {
  const user = await requireUser();
  const id = await createWork(user.id, templateId, about);
  revalidatePath("/journal");
  redirect(`/journal/w/${id}`);
}

export type SaveResult = { error: string } | { ok: true; savedAt: string };

/** Save what is in the editor. A save that changes nothing adds nothing. */
export async function saveWorkAction(id: string, title: string, body: string): Promise<SaveResult> {
  const user = await requireUser();
  try {
    const r = await saveWork(user.id, id, title, body);
    if (r.saved) revalidatePath("/journal");
    return { ok: true, savedAt: r.savedAt };
  } catch (e) {
    return { error: (e as Error).message };
  }
}
