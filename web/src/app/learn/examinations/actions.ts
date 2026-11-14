"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { attest, closeExamination, createExamination, joinExamination } from "@/server/examination-store";

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
  revalidatePath(`/learn/examinations/${id}`);
}

