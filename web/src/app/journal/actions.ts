"use server";

import { revalidatePath } from "next/cache";
import { acceptLocalDate, clean } from "@reflect/core/journal";
import { isoDate } from "@reflect/core/person";
import { requireUser } from "@/server/session";
import { addPart, closeEntry, createEntry, entryText, keepAnswer, keepLabels, startReflection, type SelfTermRow } from "@/server/journal-store";
import { readSalience } from "@/server/reader";
import { PRIMES, decide, type Salience } from "@reflect/core/state";

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

/** The model's guess at which elements are in an entry. Shown to the person as questions; nothing is kept. */
export async function guessLabels(entryId: string): Promise<{ salience: Salience | null }> {
  const user = await requireUser();
  const text = await entryText(user.id, entryId);
  if (!text) return { salience: null };
  return { salience: await readSalience(text) };
}

type TermChoice = { quote: string; term: string; whose: "self" | "other"; modelBasis: string; basis: string | null };

/** Keep the person's labels. The person's choice always wins over the model's guess. */
export async function saveLabels(entryId: string, weights: Record<string, number> | null, chosen: string[], terms: TermChoice[], guessAt?: number): Promise<Result> {
  const user = await requireUser();
  if (guessAt !== undefined && !(typeof guessAt === "number" && guessAt > 0 && guessAt <= 1)) return { error: "Unknown element." };
  if (!chosen.every((c) => PRIMES.includes(c))) return { error: "Unknown element." };
  if (weights && !Object.entries(weights).every(([k, v]) => PRIMES.includes(k) && typeof v === "number" && v >= 0 && v <= 1)) return { error: "Unknown element." };
  const rows: SelfTermRow[] = [];
  for (const t of terms) {
    if (t.basis !== null && (!PRIMES.includes(t.basis) || t.basis === "self-differentiation")) return { error: "Unknown element." };
    if (!PRIMES.includes(t.modelBasis) || (t.whose !== "self" && t.whose !== "other")) return { error: "Unknown element." };
    const quote = clean(t.quote);
    if (!quote) continue;
    rows.push({ quote, term: clean(t.term) ?? quote, whose: t.whose, modelBasis: t.modelBasis, basis: t.basis, decision: t.basis === null ? "rejected" : t.basis === t.modelBasis ? "confirmed" : "corrected" });
  }
  try {
    await keepLabels(user.id, entryId, decide(weights, new Set(chosen), guessAt), rows);
  } catch {
    return { error: "These labels couldn't be kept." };
  }
  revalidatePath("/journal");
  return { ok: true };
}
