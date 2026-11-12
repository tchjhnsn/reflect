"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { requireUser } from "@/server/session";
import { IMPORTABLE, MAX_IMPORT_BYTES } from "@reflect/core/works";
import { createWork, importWork, saveWork } from "@/server/work-store";
import { compileEdition, holdFromWork, placeClaim } from "@/server/book-store";

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

/** Import one Markdown or text file as a work. The browser reads the file and sends its text. */
export async function importWorkAction(fileName: string, text: string): Promise<{ error: string } | { ok: true }> {
  const user = await requireUser();
  if (!IMPORTABLE.test(fileName)) return { error: `${fileName}: only Markdown and text files.` };
  if (new TextEncoder().encode(text).length > MAX_IMPORT_BYTES) return { error: `${fileName}: larger than 1 MB.` };
  try {
    await importWork(user.id, fileName, text);
  } catch (e) {
    return { error: `${fileName}: ${(e as Error).message}` };
  }
  revalidatePath("/journal");
  return { ok: true };
}

/** "I hold this": a selected passage becomes a belief in the chosen chapter of the book. */
export async function holdAction(workId: string, passage: string, domain: string): Promise<{ error: string } | { ok: true }> {
  const user = await requireUser();
  try {
    await holdFromWork(user.id, workId, passage, domain);
  } catch (e) {
    return { error: (e as Error).message };
  }
  revalidatePath("/journal/book");
  return { ok: true };
}

/** Place a belief in a chapter of the book. */
export async function placeAction(claimId: string, domain: string): Promise<{ error: string } | { ok: true }> {
  const user = await requireUser();
  try {
    await placeClaim(user.id, claimId, domain);
  } catch (e) {
    return { error: (e as Error).message };
  }
  revalidatePath("/journal/book");
  return { ok: true };
}

/** Keep the book as it stands as a dated edition in the journal, and open it. */
export async function compileEditionAction(): Promise<void> {
  const user = await requireUser();
  const id = await compileEdition(user.id);
  revalidatePath("/journal");
  redirect(`/journal/w/${id}`);
}
