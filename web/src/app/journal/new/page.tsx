import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, FileIcon } from "lucide-react";
import { KINDS, TEMPLATES } from "@reflect/core/works";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { createWorkAction } from "../actions";

export const metadata: Metadata = { title: "Create" };

/** The prompt library: start blank, or from a template for a kind of thinking. */
export default async function NewWorkPage({ searchParams }: { searchParams: Promise<{ about?: string }> }) {
  await requireUser();
  const about = (await searchParams).about?.slice(0, 500) || null;
  return (
    <main className="mx-auto max-w-4xl px-6 py-6">
      <header className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
        <h1 className="text-2xl font-semibold tracking-tight">Create</h1>
      </header>
      {about ? <blockquote className="text-muted-foreground mt-6 border-l-2 pl-3 text-sm">Writing about: {about}</blockquote> : null}
      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {TEMPLATES.map((t) => (
          <form key={t.id} action={createWorkAction.bind(null, t.id, about)}>
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
