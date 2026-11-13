"use server";

import { revalidatePath } from "next/cache";
import { acceptLocalDate, clean } from "@reflect/core/journal";
import { isoDate } from "@reflect/core/person";
import { requireUser } from "@/server/session";
import { addPart, closeEntry, createEntry, keepAnswer, startReflection } from "@/server/journal-store";

type Result = { error: string } | { ok: true };

export async function post(text: string, localDate: string, promptId: string | null): Promise<Result> {
  const user = await requireUser();
  const words = clean(text);
  if (!words) return { error: "Write something first." };
  const day = isoDate(localDate);
  if (!day || !acceptLocalDate(day, new Date())) return { error: "Your device's date looks off." };
  await createEntry(user.id, words, day, promptId);
  revalidatePath("/journal");
  return { ok: true };
}

export async function more(entryId: string, text: string): Promise<Result> {
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

export async function done(entryId: string): Promise<Result> {
  const user = await requireUser();
  await closeEntry(user.id, entryId);
  revalidatePath("/journal");
  return { ok: true };
}

export async function beginReflection(entryId: string, frame: string): Promise<{ error: string } | { id: string }> {
  const user = await requireUser();
  try {
    return { id: await startReflection(user.id, entryId, frame) };
  } catch {
    return { error: "This entry can't be reflected on." };
  }
}

/** Keep one answer. An empty answer is a skip and writes nothing. */
export async function reflectAnswer(reflectionId: string, seq: number, text: string): Promise<Result> {
  const user = await requireUser();
  const words = clean(text);
  if (words === null) return { ok: true };
  if (!Number.isInteger(seq) || seq < 0) return { error: "Unknown question." };
  try {
    await keepAnswer(user.id, reflectionId, seq, words);
  } catch {
    return { error: "That answer couldn't be kept." };
  }
  revalidatePath("/journal");
  return { ok: true };
}
