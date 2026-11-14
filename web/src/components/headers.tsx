"use client";

import { Suspense, useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { ArrowLeftIcon, ChevronRightIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { BackButton } from "./back-button";
import { SearchButton } from "./search-button";

// Three kinds of header, each matched to its page (Torian, 2026-11-14): a section's pages keep the section's tabs;
// one thing's page (a book, a course, an author) says where it sits and, once its title scrolls away, carries its
// name and main action; a page where you are doing something (a lecture, the reader) is quiet, with back to the thing.
// All stay at the top of the screen and all have search.

export type Crumb = { label: string; href?: string };

const BAR = "bg-background/95 supports-[backdrop-filter]:bg-background/80 sticky top-0 z-30 backdrop-blur";

function Back({ fallback, label }: { fallback: string; label: string }) {
  return (
    <Suspense fallback={<Button asChild variant="ghost" size="icon" aria-label={label}><Link href={fallback}><ArrowLeftIcon /></Link></Button>}>
      <BackButton fallback={fallback} label={label} />
    </Suspense>
  );
}

/** A section's header: its title, its tabs (in a row of their own on a phone, which scrolls), and search. */
export function SectionHeader({ back, backLabel, title, tabs }: { back: string; backLabel: string; title: string; tabs?: ReactNode }) {
  return (
    <header className={cn(BAR, "py-2")}>
      <div className="flex min-h-10 items-center gap-3">
        <Back fallback={back} label={backLabel} />
        <h1 className="truncate text-2xl font-semibold tracking-tight">{title}</h1>
        <div className="ml-auto flex items-center gap-2">
          {tabs ? <div className="hidden items-center gap-2 md:flex">{tabs}</div> : null}
          <SearchButton />
        </div>
      </div>
      {tabs ? <div className="-mx-1 mt-2 flex items-center gap-2 overflow-x-auto px-1 pb-1 md:hidden">{tabs}</div> : null}
    </header>
  );
}

function Crumbs({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav aria-label="Breadcrumb" className="min-w-0">
      <ol className="text-muted-foreground flex min-w-0 items-center gap-1 text-sm">
        {crumbs.map((c, i) => (
          <li key={i} className={cn("flex min-w-0 items-center gap-1", i < crumbs.length - 2 ? "hidden sm:flex" : "")}>
            {c.href ? <Link href={c.href} className="hover:text-foreground truncate">{c.label}</Link> : <span className="truncate">{c.label}</span>}
            {i < crumbs.length - 1 ? <ChevronRightIcon className="size-3.5 shrink-0" /> : null}
          </li>
        ))}
      </ol>
    </nav>
  );
}

/**
 * One thing's header: back, where it sits (Learn / Browse / Courses, or Learn / My shelves / Teaching), and search.
 * Once the page's title (the element with id `page-title`) has scrolled away, it shows the thing's name and its
 * main action instead.
 */
export function ThingHeader({ back, backLabel = "Back", crumbs, title, action }: { back: string; backLabel?: string; crumbs: Crumb[]; title: string; action?: ReactNode }) {
  const [past, setPast] = useState(false);
  useEffect(() => {
    const el = document.getElementById("page-title");
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setPast(!e!.isIntersecting && e!.boundingClientRect.top < 0), { rootMargin: "-56px 0px 0px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <header className={cn(BAR, "py-2", past ? "border-b" : "")}>
      <div className="flex min-h-10 items-center gap-2">
        <Back fallback={back} label={backLabel} />
        {past ? <p className="min-w-0 truncate font-semibold">{title}</p> : <Crumbs crumbs={crumbs} />}
        <div className="ml-auto flex shrink-0 items-center gap-2">
          {past && action ? <div className="flex items-center gap-2 [&_form]:flex [&_form]:gap-2">{action}</div> : null}
          <SearchButton />
        </div>
      </div>
    </header>
  );
}

/** The header of a page where you are doing something: back to the thing, its name and where you are in it, and only what the doing needs. */
export function TaskHeader({ back, title, position, right }: { back: { href: string; label: string }; title: string; position?: string; right?: ReactNode }) {
  return (
    <header className={cn(BAR, "py-2")}>
      <div className="flex min-h-10 items-center gap-2">
        <Button asChild variant="ghost" size="icon" aria-label={back.label}><Link href={back.href}><ArrowLeftIcon /></Link></Button>
        <Link href={back.href} className="min-w-0 truncate text-sm hover:underline"><span className="font-medium">{title}</span>{position ? <span className="text-muted-foreground"> · {position}</span> : null}</Link>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {right}
          <SearchButton />
        </div>
      </div>
    </header>
  );
}
