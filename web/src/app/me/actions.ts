"use server";

import { revalidatePath } from "next/cache";
import { acceptLocalDate, clean } from "@/core/journal";
import { isoDate, type IsoDate } from "@/core/person";
import { requireUser } from "@/server/session";
import { addBelief, choosePack, isFactKey, letGo, stateFactNow, stateFriendNow } from "@/server/me-store";

type Result = { error: string } | { ok: true };

function today(localDate: string): IsoDate | null {
  const d = isoDate(localDate);
  return d && acceptLocalDate(d, new Date()) ? d : null;
}

function done(): Result {
  revalidatePath("/me");
  revalidatePath("/journal");
  return { ok: true };
}

/** Change or remove one thing about you. The old value keeps its dates. */
export async function changeFact(key: string, text: string, localDate: string): Promise<Result> {
  const user = await requireUser();
  const at = today(localDate);
  if (!at) return { error: "Your device's date looks off." };
  const value = clean(text) || null;
  if (key === "closest_friend") {
    await stateFriendNow(user.id, value, at);
    return done();
  }
  if (!isFactKey(key)) return { error: "Unknown field." };
  if (key === "name" && value === null) return { error: "Keep a name, even a nickname." };
  if (key === "date_of_birth" && value !== null) {
    const d = isoDate(value);
    if (!d) return { error: "Write the date as YYYY-MM-DD." };
    if (d > at) return { error: "A date of birth can't be in the future." };
  }
  await stateFactNow(user.id, key, value, at);
  return done();
}

export async function stateBelief(kind: string, text: string, localDate: string): Promise<Result> {
  const user = await requireUser();
  const at = today(localDate);
  if (!at) return { error: "Your device's date looks off." };
  if (kind !== "belief" && kind !== "value") return { error: "Unknown kind." };
  const words = clean(text);
  if (!words) return { error: "Write it in your own words first." };
  await addBelief(user.id, kind, words, at);
  return done();
}

export async function letGoBelief(id: string, localDate: string): Promise<Result> {
  const user = await requireUser();
  const at = today(localDate);
  if (!at) return { error: "Your device's date looks off." };
  await letGo(user.id, id, at);
  return done();
}

export async function setPack(packId: string, chosen: boolean): Promise<Result> {
  const user = await requireUser();
  try {
    await choosePack(user.id, packId, chosen);
  } catch {
    return { error: "Unknown pack." };
  }
  return done();
}
