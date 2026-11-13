"use client";

import { useState } from "react";
import { BookOpenIcon, XIcon } from "lucide-react";
import { readerUrl } from "@reflect/core/googlebooks";
import { Button } from "@/components/ui/button";

/**
 * Google's reader for the book, opened on the book page when the person asks for it: the whole book when it is
 * free, a preview otherwise. The reader's own search box searches inside the book.
 */
export function Reader({ id, full }: { id: string; full: boolean }) {
  const [open, setOpen] = useState(false);
  const label = full ? "Read it free" : "Read a preview";
  if (!open) {
    return (
      <Button variant="outline" onClick={() => setOpen(true)}>
        <BookOpenIcon />
        {label}
      </Button>
    );
  }
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-muted-foreground text-xs">{full ? "The whole book, free" : "A preview"}, from Google Books. Search inside it with the box at its top.</p>
        <Button variant="ghost" size="sm" onClick={() => setOpen(false)}><XIcon />Close</Button>
      </div>
      <iframe src={readerUrl(id)} title={`${label}, from Google Books`} className="bg-muted h-[70vh] min-h-96 w-full rounded-md border" />
    </section>
  );
}
