"use server";

import { revalidatePath } from "next/cache";
import { cookies } from "next/headers";
import { THEMES, THEME_COOKIE } from "@reflect/core/theme";
import { acceptLocalDate, clean } from "@reflect/core/journal";
import { isoDate, type IsoDate } from "@reflect/core/person";
import { requireUser } from "@/server/session";
import { setResearchAllowed } from "@/server/journal-store";
import { addBelief, choosePack, isFactKey, isHoldSubject, letGo, stateFactNow, stateFriendNow, stateHoldNow } from "@/server/me-store";
import { CHOICES, cleanHold } from "@reflect/core/onboarding";

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
  if (key.startsWith("hold_")) {
    const subject = key.slice(5);
    if (!isHoldSubject(subject)) return { error: "Unknown field." };
    if (value === null) {
      await stateHoldNow(user.id, subject, null, at);
      return done();
    }
    const r = cleanHold(value);
    if ("error" in r) return { error: r.error };
    await stateHoldNow(user.id, subject, r.value, at);
    return done();
  }
  if (key === "closest_friend") {
    await stateFriendNow(user.id, value, at);
    return done();
  }
  if (!isFactKey(key)) return { error: "Unknown field." };
  if (key === "name" && value === null) return { error: "Keep a name, even a nickname." };
  if (value !== null && CHOICES[key] && !CHOICES[key]!.includes(value)) return { error: "Pick one." };
  if (key === "children" && value !== null && !(/^\d+$/.test(value) && Number(value) <= 30)) return { error: "Write a number." };
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

/** The look is a preference on this device, kept in a cookie. Nothing about it reaches the journal. */
export async function setTheme(theme: string): Promise<Result> {
  await requireUser();
  if (!THEMES.some((t) => t.id === theme)) return { error: "Unknown theme." };
  (await cookies()).set(THEME_COOKIE, theme, { path: "/", maxAge: 60 * 60 * 24 * 365, sameSite: "lax", secure: process.env.NODE_ENV === "production" });
  revalidatePath("/", "layout");
  return { ok: true };
}

/** Whether the person's label corrections may be used to improve the reader. Their words never are. */
export async function setResearch(allowed: boolean): Promise<Result> {
  const user = await requireUser();
  if (typeof allowed !== "boolean") return { error: "Pick on or off." };
  await setResearchAllowed(user.id, allowed);
  revalidatePath("/me");
  return { ok: true };
}
