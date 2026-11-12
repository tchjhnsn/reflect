"use server";

import { revalidatePath } from "next/cache";
import { acceptLocalDate, clean } from "@reflect/core/journal";
import { isoDate } from "@reflect/core/person";
import { requireUser } from "@/server/session";
import { addPart, closeEntry, createEntry } from "@/server/journal-store";

export type Result = { error: string } | { ok: true };

/** A new entry, in the person's words, dated by their device. */
export async function writeEntry(text: string, localDate: string): Promise<Result> {
  const user = await requireUser();
  const words = clean(text);
  if (!words) return { error: "Write something first." };
  const day = isoDate(localDate);
  if (!day || !acceptLocalDate(day, new Date())) return { error: "Your device's date looks off." };
  await createEntry(user.id, words, day, null);
  revalidatePath("/journal");
  return { ok: true };
}

/** More words on an open entry. */
export async function addToEntry(entryId: string, text: string): Promise<Result> {
  const user = await requireUser();
  const words = clean(text);
  if (!words) return { error: "Write something first." };
  try {
    await addPart(user.id, entryId, words);
  } catch {
    return { error: "This entry is closed." };
  }
  revalidatePath("/journal");
  return { ok: true };
}

/** Close an entry: nothing more is added to it. */
export async function closeEntryAction(entryId: string): Promise<Result> {
  const user = await requireUser();
  await closeEntry(user.id, entryId);
  revalidatePath("/journal");
  return { ok: true };
}
