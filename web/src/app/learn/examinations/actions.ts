"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { attest, closeExamination, createExamination, examinationAsExaminer, getExamination, joinExamination } from "@/server/examination-store";
import { callFor, endCall } from "@/server/video";

const str = (f: FormData, k: string) => String(f.get(k) ?? "");

export async function createExaminationAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const { id } = await createExamination(user.id, { courseUrl: str(form, "url"), courseTitle: str(form, "title"), name: form.get("name"), claim: form.get("claim") });
  redirect(`/learn/examinations/${id}`);
}

export async function joinExaminationAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const token = str(form, "token");
  const r = await joinExamination(user.id, token, form.get("name"));
  if ("refused" in r && r.refused === "own") redirect("/learn");
  revalidatePath(`/learn/examine/${token}`);
}

export async function attestAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await attest(user.id, str(form, "id"), str(form, "attests") === "yes", form.get("words"));
  revalidatePath(`/learn/examine/${str(form, "token")}`);
}

export async function closeExaminationAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = str(form, "id");
  await closeExamination(user.id, id);
  await endCall(id);
  revalidatePath(`/learn/examinations/${id}`);
}

/** A place in the examination's call, for its candidate or one of its examiners only. */
export async function examCallAction(id: string): Promise<{ url: string } | null> {
  const user = await requireUser();
  const mine = await getExamination(user.id, String(id));
  if (mine) return mine.status === "open" ? callFor(mine.id, { name: mine.name, owner: true }) : null;
  const theirs = await examinationAsExaminer(user.id, String(id));
  const me = theirs?.examiners.find((x) => x.userId === user.id);
  return theirs && me && theirs.status === "open" ? callFor(theirs.id, { name: me.name, owner: false }) : null;
}
