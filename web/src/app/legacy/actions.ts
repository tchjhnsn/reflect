"use server";

import { revalidatePath } from "next/cache";
import { requireUser } from "@/server/session";
import { confirmDeath, stopRelease } from "@/server/legacy-store";

/** The person stops a release that has not happened. */
export async function stopMine(): Promise<void> {
  const user = await requireUser();
  await stopRelease({ userId: user.id });
  revalidatePath("/legacy");
}

/** A trustee confirms the death; the form makes them type that they mean it. */
export async function confirmAsTrustee(form: FormData): Promise<void> {
  const token = String(form.get("token") ?? "");
  if (String(form.get("sure") ?? "") !== "on") return;
  await confirmDeath(token);
  revalidatePath(`/legacy/trustee/${token}`);
}

export async function stopAsTrustee(form: FormData): Promise<void> {
  const token = String(form.get("token") ?? "");
  await stopRelease({ trusteeToken: token });
  revalidatePath(`/legacy/trustee/${token}`);
}
