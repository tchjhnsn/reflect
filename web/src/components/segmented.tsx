import Link from "next/link";
import { cn } from "@/lib/utils";

/** Links drawn as shadcn's tabs list: one per view, the current one raised. Views live in the URL. */
export function Segmented({ label, items }: { label: string; items: { label: string; href: string; on: boolean }[] }) {
  return (
    <nav aria-label={label} className="bg-muted text-muted-foreground inline-flex h-9 items-center rounded-lg p-[3px]">
      {items.map((it) => (
        <Link
          key={it.href}
          href={it.href}
          aria-current={it.on ? "page" : undefined}
          className={cn("rounded-md px-3 py-1 text-sm font-medium whitespace-nowrap transition", it.on ? "bg-background text-foreground shadow-sm" : "hover:text-foreground")}
        >
          {it.label}
        </Link>
      ))}
    </nav>
  );
}
