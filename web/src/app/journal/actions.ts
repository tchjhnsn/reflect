"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { requireUser } from "@/server/session";
import { IMPORTABLE, MAX_IMPORT_BYTES } from "@reflect/core/works";
import type { Holder } from "@reflect/core/claims";
import { createWork, fileWork, importWork, saveWork } from "@/server/work-store";
import { placeSubject } from "@/server/subject-domain-store";
import { claimFromWork, beliefFromWork, compileEdition, markBelief, placeClaim, type Marking } from "@/server/book-store";

/** Start a work from a template of the prompt library, filed in a domain when started from one, and open it. */
export async function createWorkAction(templateId: string, about: string | null, domain: string | null = null): Promise<void> {
  const user = await requireUser();
  const id = await createWork(user.id, templateId, about, domain);
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

type Done = { error: string } | { ok: true };
const attempt = async (run: () => Promise<unknown>, paths: string[]): Promise<Done> => {
  try {
    await run();
  } catch (e) {
    return { error: (e as Error).message };
  }
  for (const p of paths) revalidatePath(p, p.includes("[") ? "page" : undefined);
  return { ok: true };
};

/** A selected passage becomes a belief: an ideal or a practical belief, in the chosen domain, with its origins and holders. */
export async function beliefFromWorkAction(workId: string, passage: string, domain: string, m: Marking & { holders?: Holder[] }): Promise<Done> {
  const user = await requireUser();
  return attempt(() => beliefFromWork(user.id, workId, passage, domain, m), ["/journal/book", "/journal/d/[domain]"]);
}

/** Send a selected passage to Learn as a claim under a subject, with the person's stance. */
export async function sendToLearnAction(workId: string, passage: string, subject: string, stance: string): Promise<Done> {
  const user = await requireUser();
  return attempt(() => claimFromWork(user.id, workId, passage, subject, stance), ["/learn", "/journal/book", "/journal/d/[domain]"]);
}

/** Mark a belief already kept an ideal or a practical belief, with its origins. */
export async function markBeliefAction(claimId: string, m: Marking): Promise<Done> {
  const user = await requireUser();
  return attempt(() => markBelief(user.id, claimId, m), ["/journal/book", "/journal/d/[domain]"]);
}

/** File a work in a domain, or in none. */
export async function fileWorkAction(workId: string, domain: string | null): Promise<Done> {
  const user = await requireUser();
  return attempt(() => fileWork(user.id, workId, domain), ["/journal", "/journal/d/[domain]"]);
}

/** Place a Learn subject in a domain, or take it out of every domain (null). */
export async function placeSubjectAction(subjectId: string, domain: string | null): Promise<Done> {
  const user = await requireUser();
  return attempt(() => placeSubject(user.id, subjectId, domain), ["/journal/book", "/journal/d/[domain]"]);
}

/** Place a belief in a chapter of the book. */
export async function placeAction(claimId: string, domain: string): Promise<{ error: string } | { ok: true }> {
  const user = await requireUser();
  return attempt(() => placeClaim(user.id, claimId, domain), ["/journal/book", "/journal/d/[domain]"]);
}

/** Keep the book as it stands as a dated edition in the journal, and open it. */
export async function compileEditionAction(): Promise<void> {
  const user = await requireUser();
  const id = await compileEdition(user.id);
  revalidatePath("/journal");
  redirect(`/journal/w/${id}`);
}
