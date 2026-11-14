"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { parseRef, refHref, refString } from "@reflect/core/work-browse";
import { requireUser } from "@/server/session";
import { addLifeRecord, lifeRecords, saveLifeRecord } from "@/server/life-store";

/**
 * Keeps a job, company or imagined work as an inspiration, with what draws the person to it, the origins they mark
 * and for the sake of what. Keeping it again changes the one already kept.
 */
export async function inspireAction(ref: string, title: string, form: FormData): Promise<void> {
  const user = await requireUser();
  const r = parseRef(ref);
  if (!r) throw new Error("Nothing to keep.");
  const type = r.kind === "job" || r.kind === "past" ? "job" : r.kind === "company" ? "company" : "other";
  const body = { ref: refString(r), title: title.slice(0, 300), type, draws: String(form.get("draws") ?? ""), origins: form.getAll("origins").map(String).join(","), end: String(form.get("end") ?? "") };
  const had = (await lifeRecords(user.id, "work")).find((x) => x.kind === "inspiration" && x.body.ref === body.ref);
  // Keep what the form does not carry (what is learned toward it, its kind of work when the person set it).
  if (had) await saveLifeRecord(user.id, had.id, { ...had.body, ...body, type: had.body.type ?? type });
  else await addLifeRecord(user.id, "work", "inspiration", body);
  revalidatePath("/", "layout");
  redirect(`${refHref(r)}#inspires`);
}
