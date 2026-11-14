"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { addLifeRecord, keepLifeAct, lifeRecord, lifeRecords, saveLifeRecord } from "@/server/life-store";
import { createWork } from "@/server/work-store";
import { listItems } from "@/server/ways-store";

const TYPES = new Set(["job", "career", "venture", "company", "other"]);

/** Add work the person wants that is not in a catalog: a job, a career or a venture of their own, by name. */
export async function addOwnAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const title = String(form.get("title") ?? "").trim().slice(0, 300);
  if (!title) throw new Error("Name the work.");
  const type = String(form.get("type") ?? "");
  const id = await addLifeRecord(user.id, "work", "inspiration", { title, type: TYPES.has(type) ? type : "other" });
  revalidatePath("/", "layout");
  redirect(`/life/work/${id}`);
}

/** Start a journal entry toward work the person wants, kept on that work as what moved them toward it. */
export async function writeTowardAction(recordId: string): Promise<void> {
  const user = await requireUser();
  const r = await lifeRecord(user.id, recordId);
  if (!r || r.app !== "work") throw new Error("No such work.");
  const title = String(r.kind === "imagined" ? r.body.name ?? "" : r.body.title ?? "");
  const id = await createWork(user.id, "blank", `What moved me toward ${title}`);
  await keepLifeAct(user.id, recordId, { kind: "wrote", day: new Date().toISOString().slice(0, 10), note: id });
  revalidatePath("/", "layout");
  redirect(`/journal/w/${id}`);
}

/** Say which of the things being learned in Learn are toward this work. */
export async function learningTowardAction(recordId: string, back: string, form: FormData): Promise<void> {
  const user = await requireUser();
  const r = await lifeRecord(user.id, recordId);
  if (!r || r.app !== "work") throw new Error("No such work.");
  const mine = new Set((await listItems(user.id)).map((i) => i.id));
  const ids = form.getAll("learning").map(String).filter((id) => mine.has(id));
  await saveLifeRecord(user.id, recordId, { ...r.body, learning: ids.join(",") });
  revalidatePath("/", "layout");
  redirect(back.startsWith("/life/work") ? `${back}#toward` : "/life/work");
}

/** The hours a week the person can give to work right now. */
export async function hoursAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const raw = String(form.get("hours") ?? "").trim();
  const hours = raw === "" ? undefined : Math.round(Number(raw));
  if (hours !== undefined && !(hours >= 0 && hours <= 100)) throw new Error("Give hours from 0 to 100.");
  const had = (await lifeRecords(user.id, "work")).find((r) => r.kind === "now");
  if (had) await saveLifeRecord(user.id, had.id, { hours });
  else await addLifeRecord(user.id, "work", "now", { hours });
  revalidatePath("/", "layout");
  redirect("/life/work#now");
}
