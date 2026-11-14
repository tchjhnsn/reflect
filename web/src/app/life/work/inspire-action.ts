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
  const body = { ref: refString(r), title: title.slice(0, 300), draws: String(form.get("draws") ?? ""), origins: form.getAll("origins").map(String).join(","), end: String(form.get("end") ?? "") };
  const had = (await lifeRecords(user.id, "work")).find((x) => x.kind === "inspiration" && x.body.ref === body.ref);
  if (had) await saveLifeRecord(user.id, had.id, body);
  else await addLifeRecord(user.id, "work", "inspiration", body);
  revalidatePath("/", "layout");
  redirect(`${refHref(r)}#inspires`);
}
