"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeftIcon, BoldIcon, BookmarkCheckIcon, Heading2Icon, ItalicIcon, ListIcon, QuoteIcon } from "lucide-react";
import { DOMAINS } from "@reflect/core/book";
import { EditorContent, useEditor, useEditorState } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Markdown } from "@tiptap/markdown";
import Placeholder from "@tiptap/extension-placeholder";
import { KINDS } from "@reflect/core/works";
import type { Work } from "@/server/work-store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { holdAction, saveWorkAction } from "../../actions";

type Status = "saved" | "unsaved" | "saving" | "failed";
const SAVE_AFTER_MS = 1200;

/**
 * A page to write on. The body is Markdown (so a work leaves as a plain file); headings, lists, quotes, bold
 * and italic come from typing Markdown or from the small bar. It saves on its own a moment after you stop
 * typing, and each save that changes something is kept as a new version.
 */
export function Editor({ work }: { work: Work }) {
  const [title, setTitle] = useState(work.title);
  const [status, setStatus] = useState<Status>("saved");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const titleRef = useRef(work.title);
  const saving = useRef(false);
  const scheduleRef = useRef<() => void>(() => undefined);

  const editor = useEditor({
    immediatelyRender: false,
    extensions: [StarterKit.configure({ link: { openOnClick: false } }), Markdown, Placeholder.configure({ placeholder: "Write..." })],
    content: work.body,
    contentType: "markdown",
    editorProps: { attributes: { class: "writing min-h-[60vh]", "aria-label": "Your writing" } },
    // The editor keeps its first options, so it calls the latest schedule through a ref.
    onUpdate: () => scheduleRef.current(),
  });

  const save = useCallback(async () => {
    if (!editor || saving.current) return;
    saving.current = true;
    setStatus("saving");
    const r = await saveWorkAction(work.id, titleRef.current, editor.getMarkdown());
    saving.current = false;
    setStatus("error" in r ? "failed" : "saved");
  }, [editor, work.id]);

  const schedule = useCallback(() => {
    setStatus("unsaved");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => void save(), SAVE_AFTER_MS);
  }, [save]);
  useEffect(() => {
    scheduleRef.current = schedule;
  }, [schedule]);

  // Save before leaving the page, and on Cmd/Ctrl+S.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        if (timer.current) clearTimeout(timer.current);
        void save();
      }
    };
    const leave = (e: BeforeUnloadEvent) => {
      if (status === "unsaved" || status === "saving") e.preventDefault();
    };
    window.addEventListener("keydown", key);
    window.addEventListener("beforeunload", leave);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("beforeunload", leave);
    };
  }, [save, status]);

  const marks = useEditorState({
    editor,
    selector: ({ editor: e }) => ({
      bold: e?.isActive("bold") ?? false,
      italic: e?.isActive("italic") ?? false,
      h2: e?.isActive("heading", { level: 2 }) ?? false,
      list: e?.isActive("bulletList") ?? false,
      quote: e?.isActive("blockquote") ?? false,
      selected: e && !e.state.selection.empty ? e.state.doc.textBetween(e.state.selection.from, e.state.selection.to, " ").trim() : "",
    }),
  });

  // "I hold this": the selected passage becomes a belief, placed in a chapter of the book.
  const [hold, setHold] = useState<{ passage: string; note?: string; error?: string } | null>(null);
  const [holding, setHolding] = useState(false);
  const keepHold = async (domain: string) => {
    if (!hold) return;
    setHolding(true);
    const r = await holdAction(work.id, hold.passage, domain);
    setHolding(false);
    if ("error" in r) setHold({ ...hold, error: r.error });
    else setHold({ passage: hold.passage, note: DOMAINS.find((d) => d.id === domain)!.title });
  };

  const tool = (label: string, Icon: typeof BoldIcon, on: boolean | undefined, run: () => void) => (
    <Button key={label} type="button" size="icon" variant="ghost" className={cn("size-8", on && "bg-muted")} aria-label={label} aria-pressed={!!on} onMouseDown={(e) => e.preventDefault()} onClick={run}>
      <Icon className="size-4" />
    </Button>
  );

  return (
    <main className="min-h-screen">
      <header className="bg-background/90 sticky top-0 z-10 border-b backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-2 px-4 py-2">
          <Button asChild variant="ghost" size="icon" aria-label="Journal" onClick={() => void save()}><Link href="/journal"><ArrowLeftIcon /></Link></Button>
          <span className="text-muted-foreground hidden truncate text-sm sm:inline">{KINDS[work.kind].label}</span>
          <div className="ml-auto flex items-center gap-0.5">
            {editor ? [
              tool("Heading", Heading2Icon, marks?.h2, () => editor.chain().focus().toggleHeading({ level: 2 }).run()),
              tool("Bold", BoldIcon, marks?.bold, () => editor.chain().focus().toggleBold().run()),
              tool("Italic", ItalicIcon, marks?.italic, () => editor.chain().focus().toggleItalic().run()),
              tool("List", ListIcon, marks?.list, () => editor.chain().focus().toggleBulletList().run()),
              tool("Quote", QuoteIcon, marks?.quote, () => editor.chain().focus().toggleBlockquote().run()),
            ] : null}
            <Button type="button" size="sm" variant="ghost" className="ml-1 h-8" disabled={!marks?.selected} onMouseDown={(e) => e.preventDefault()} onClick={() => setHold({ passage: marks!.selected })}>
              <BookmarkCheckIcon className="size-4" /><span className="hidden sm:inline">I hold this</span>
            </Button>
            <span className={cn("text-muted-foreground ml-3 w-16 text-right text-xs", status === "failed" && "text-destructive")} role="status">
              {status === "saved" ? "Saved" : status === "saving" ? "Saving" : status === "failed" ? "Not saved" : "Edited"}
            </span>
          </div>
        </div>
        {hold ? (
          <div className="mx-auto max-w-3xl space-y-2 px-6 pb-3">
            <blockquote className="border-l-2 pl-3 font-serif text-base italic">{hold.passage}</blockquote>
            {hold.note ? (
              <p className="text-sm">Kept in {hold.note}. <Link href="/journal/book" className="underline underline-offset-2">See the book</Link> <button type="button" className="text-muted-foreground ml-2 text-xs" onClick={() => setHold(null)}>Close</button></p>
            ) : (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-muted-foreground mr-1 text-sm">Which chapter of your life?</span>
                {DOMAINS.map((d) => <Button key={d.id} type="button" size="sm" variant="outline" className="h-7" disabled={holding} onClick={() => void keepHold(d.id)}>{d.title}</Button>)}
                <Button type="button" size="sm" variant="ghost" className="h-7" onClick={() => setHold(null)}>Cancel</Button>
                {hold.error ? <span className="text-destructive text-sm" role="alert">{hold.error}</span> : null}
              </div>
            )}
          </div>
        ) : null}
      </header>
      <article className="mx-auto max-w-2xl px-6 pt-12 pb-32">
        <input
          value={title}
          onChange={(e) => {
            setTitle(e.target.value);
            titleRef.current = e.target.value;
            schedule();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              if (!editor) return;
              // Into the first empty line (under a template's first heading), else the start.
              let at: number | null = null;
              editor.state.doc.descendants((n, pos) => {
                if (at === null && n.type.name === "paragraph" && n.content.size === 0) at = pos + 1;
                return at === null;
              });
              editor.view.focus();
              editor.commands.setTextSelection(at ?? 1);
            }
          }}
          placeholder="Untitled"
          aria-label="Title"
          className="placeholder:text-muted-foreground/50 mb-6 w-full bg-transparent font-serif text-4xl font-semibold tracking-tight outline-none"
        />
        <EditorContent editor={editor} />
      </article>
    </main>
  );
}
