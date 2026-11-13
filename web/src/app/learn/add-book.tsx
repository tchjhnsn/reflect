"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { CheckIcon, ChevronDownIcon } from "lucide-react";
import type { CatalogEntry } from "@reflect/core/openlibrary";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { addFromLibraryAction } from "./actions";

const LABEL = { want: "Want to read", reading: "Reading", read: "Read" } as const;

/** Put a book from the open library on your shelves: want to read by default, or reading or read. */
export function AddBook({ entry }: { entry: CatalogEntry }) {
  const [added, setAdded] = useState<keyof typeof LABEL | null>(null);
  const [error, setError] = useState(false);
  const [pending, run] = useTransition();
  const router = useRouter();
  const add = (status: keyof typeof LABEL) => run(async () => {
    const f = new FormData();
    f.set("entry", JSON.stringify(entry));
    f.set("status", status);
    try {
      await addFromLibraryAction(f);
      setAdded(status);
      router.refresh();
    } catch {
      setError(true);
    }
  });
  if (added) return <span className="text-muted-foreground inline-flex h-8 items-center gap-1 text-xs"><CheckIcon className="size-3.5" />{LABEL[added]}</span>;
  return (
    <div className="flex items-center">
      <Button size="sm" className="h-8 rounded-r-none" disabled={pending} onClick={() => add("want")}>{pending ? "Adding..." : "Want to read"}</Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button size="sm" className="h-8 rounded-l-none border-l border-l-white/20 px-2" disabled={pending} aria-label="Add as reading or read"><ChevronDownIcon /></Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => add("reading")}>Reading</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => add("read")}>Read</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {error ? <span className="text-destructive ml-2 text-xs" role="alert">Not added</span> : null}
    </div>
  );
}
