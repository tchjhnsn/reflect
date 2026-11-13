import type { Metadata } from "next";
import { FileIcon } from "lucide-react";
import { KINDS, TEMPLATES } from "@reflect/core/works";
import { DOMAINS, isDomain } from "@reflect/core/book";
import { PAGE, PageHeader } from "@/components/page-header";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { createWorkAction } from "../actions";

export const metadata: Metadata = { title: "Create" };

/** The prompt library: start blank, or from a template for a kind of thinking. */
export default async function NewWorkPage({ searchParams }: { searchParams: Promise<{ about?: string; domain?: string }> }) {
  await requireUser();
  const q = await searchParams;
  const about = q.about?.slice(0, 500) || null;
  const domain = isDomain(q.domain) ? q.domain : null;
  const back = domain ? `/journal/d/${domain}` : "/journal";
  return (
    <main className={PAGE}>
      <PageHeader back={back} backLabel="Back" title={<>Create{domain ? <span className="text-muted-foreground font-normal"> in {DOMAINS.find((d) => d.id === domain)!.title}</span> : null}</>}></PageHeader>
      {about ? <blockquote className="text-muted-foreground mt-6 border-l-2 pl-3 text-sm">Writing about: {about}</blockquote> : null}
      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {TEMPLATES.map((t) => (
          <form key={t.id} action={createWorkAction.bind(null, t.id, about, domain)}>
            <button type="submit" className="group h-full w-full text-left">
              <Card className="group-hover:border-foreground/30 h-full gap-2 py-5 transition-colors">
                <CardHeader className="px-5">
                  {t.id === "blank" ? <FileIcon className="text-muted-foreground mb-1 size-5" aria-hidden="true" /> : <CardDescription className="text-xs">{KINDS[t.kind].label}</CardDescription>}
                  <CardTitle className="text-base">{t.name}</CardTitle>
                  <CardDescription>{t.about}</CardDescription>
                  {t.source ? <p className="text-muted-foreground/80 pt-1 text-xs italic">{t.source}</p> : null}
                </CardHeader>
              </Card>
            </button>
          </form>
        ))}
      </section>
    </main>
  );
}
