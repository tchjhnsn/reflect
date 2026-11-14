import type { ReactNode } from "react";
import { Suspense } from "react";
import Link from "next/link";
import { ArrowLeftIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BackButton } from "./back-button";
import { SearchButton } from "./search-button";

/** The page width every screen shares, so the header sits in the same place as you move between pages. */
export const PAGE = "mx-auto w-full max-w-5xl px-6 py-6";

/**
 * The header of the journal's, the log's and the chat's pages (Learn's are in `./headers`): back at the left (the page you came from, else `back`), the title beside it, the page's actions at the right.
 * Same height, type and spacing everywhere, so only the words change as you move around.
 */
export function PageHeader({ back, backLabel, title, children }: { back: string; backLabel: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="bg-background/95 supports-[backdrop-filter]:bg-background/80 sticky top-0 z-30 flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-2 py-2 backdrop-blur">
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <Suspense fallback={<Button asChild variant="ghost" size="icon" aria-label={backLabel}><Link href={back}><ArrowLeftIcon /></Link></Button>}><BackButton fallback={back} label={backLabel} /></Suspense>
        <h1 className="min-w-0 flex-1 truncate text-2xl font-semibold tracking-tight">{title}</h1>
        <span className="sm:hidden"><SearchButton keys={false} /></span>
      </div>
      <div className="flex flex-wrap items-center gap-2">{children}<span className="hidden sm:inline-flex"><SearchButton /></span></div>
    </header>
  );
}
