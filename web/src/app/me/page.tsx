import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { hasOnboarded } from "@/server/person-store";
import { loadMe } from "@/server/me-store";
import { researchAllowed } from "@/server/journal-store";
import { cookies } from "next/headers";
import { THEME_COOKIE, asTheme } from "@/core/theme";
import { MePage } from "./me";

export const metadata: Metadata = { title: "Me" };
export const dynamic = "force-dynamic";

export default async function Page() {
  const user = await requireUser();
  if (!(await hasOnboarded(user.id))) redirect("/welcome");
  const theme = asTheme((await cookies()).get(THEME_COOKIE)?.value);
  const [me, research] = await Promise.all([loadMe(user.id), researchAllowed(user.id)]);
  return <MePage me={me} theme={theme} research={research} />;
}
