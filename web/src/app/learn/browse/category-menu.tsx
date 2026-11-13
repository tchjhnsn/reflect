"use client";

import Link from "next/link";
import { CheckIcon, ChevronDownIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export type MenuItem = { href: string; label: string; on: boolean };

/** One quiet button that opens a list of choices, each a link; the button names the current one. */
export function Menu({ label, items, lead = 0 }: { label: string; items: MenuItem[]; /** How many items come before a separator. */ lead?: number }) {
  const current = items.find((i) => i.on)?.label ?? items[0]?.label ?? label;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="h-10 max-w-full justify-between gap-2" aria-label={`${label}: ${current}`}>
          <span className="truncate">{current}</span>
          <ChevronDownIcon className="text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-[70vh] w-64 overflow-y-auto">
        {items.map((i, n) => (
          <span key={i.href} className="contents">
            {lead && n === lead ? <DropdownMenuSeparator /> : null}
            <DropdownMenuItem asChild>
              <Link href={i.href} aria-current={i.on ? "page" : undefined} className="flex items-center justify-between gap-2">
                {i.label}
                {i.on ? <CheckIcon className="size-4" /> : null}
              </Link>
            </DropdownMenuItem>
          </span>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
