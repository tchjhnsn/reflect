import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { currentName, hasOnboarded } from "@/server/person-store";

export const metadata: Metadata = { title: "Your journal" };
export const dynamic = "force-dynamic";

export default async function JournalPage() {
  const user = await requireUser();
  if (!(await hasOnboarded(user.id))) redirect("/welcome");
  const name = await currentName(user.id);
  return (
    <main className="shell">
      <section className="screen">
        <p className="eyebrow">Reflect</p>
        <h1 className="prompt" data-testid="journal-hello">
          {name ? `Your journal starts here, ${name}.` : "Your journal starts here."}
        </h1>
      </section>
    </main>
  );
}
