"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeftIcon, BoldIcon, Heading2Icon, ItalicIcon, ListIcon, QuoteIcon } from "lucide-react";
import { DOMAINS, isDomain, type Domain } from "@reflect/core/book";
import { EditorContent, useEditor, useEditorState } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Markdown } from "@tiptap/markdown";
import Placeholder from "@tiptap/extension-placeholder";
import { KINDS } from "@reflect/core/works";
import type { Work } from "@/server/work-store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { fileWorkAction, saveWorkAction } from "../../actions";
import { Mark, type IdealOption } from "../../mark";
import { SendToLearn } from "../../send-learn";

type Status = "saved" | "unsaved" | "saving" | "failed";
const SAVE_AFTER_MS = 1200;

/**
 * A page to write on. The body is Markdown (so a work leaves as a plain file); headings, lists, quotes, bold
 * and italic come from typing Markdown or from the small bar. It saves on its own a moment after you stop
 * typing, and each save that changes something is kept as a new version.
 */
export function Editor({ work, ideals, subjects }: { work: Work; ideals: IdealOption[]; subjects: string[] }) {
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

  // The work's domain, and marking a selected passage an ideal or a practical belief.
  const [domain, setDomain] = useState<Domain | null>(work.domain);
  const file = async (d: Domain | null) => {
    setDomain(d);
    const r = await fileWorkAction(work.id, d);
    if ("error" in r) setDomain(work.domain);
  };
  const [mark, setMark] = useState<{ passage: string; kept?: Domain | null } | null>(null);
  const [send, setSend] = useState<{ passage: string; sent?: string } | null>(null);

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
          <span className="text-muted-foreground hidden truncate text-sm sm:inline">{KINDS[work.kind].label} in</span>
          <select aria-label="Domain" value={domain ?? ""} onChange={(e) => void file(isDomain(e.target.value) ? e.target.value : null)}
            className="text-muted-foreground hover:text-foreground rounded-md bg-transparent px-1 py-1 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
            <option value="">no domain</option>
            {DOMAINS.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}
          </select>
          <div className="ml-auto flex items-center gap-0.5">
            {editor ? [
              tool("Heading", Heading2Icon, marks?.h2, () => editor.chain().focus().toggleHeading({ level: 2 }).run()),
              tool("Bold", BoldIcon, marks?.bold, () => editor.chain().focus().toggleBold().run()),
              tool("Italic", ItalicIcon, marks?.italic, () => editor.chain().focus().toggleItalic().run()),
              tool("List", ListIcon, marks?.list, () => editor.chain().focus().toggleBulletList().run()),
              tool("Quote", QuoteIcon, marks?.quote, () => editor.chain().focus().toggleBlockquote().run()),
            ] : null}
            <Button type="button" size="sm" variant="ghost" className="ml-1 h-8" disabled={!marks?.selected} onMouseDown={(e) => e.preventDefault()} onClick={() => { setMark(null); setSend({ passage: marks!.selected }); }}>
              Send to Learn
            </Button>
            <Button type="button" size="sm" variant="ghost" className="h-8" disabled={!marks?.selected} onMouseDown={(e) => e.preventDefault()} onClick={() => { setSend(null); setMark({ passage: marks!.selected }); }}>
              Mark as a belief
            </Button>
            <span className={cn("text-muted-foreground ml-3 w-16 text-right text-xs", status === "failed" && "text-destructive")} role="status">
              {status === "saved" ? "Saved" : status === "saving" ? "Saving" : status === "failed" ? "Not saved" : "Edited"}
            </span>
          </div>
        </div>
        {send ? (
          <div className="mx-auto max-w-3xl space-y-3 px-6 pb-4">
            <blockquote className="border-l-2 pl-3 font-serif text-base italic">{send.passage}</blockquote>
            {send.sent !== undefined ? (
              <p className="text-sm">
                Kept in Learn under {send.sent}. <Link href="/learn?tab=claims" className="underline underline-offset-2">See your claims</Link>
                <button type="button" className="text-muted-foreground ml-2 text-xs" onClick={() => setSend(null)}>Close</button>
              </p>
            ) : (
              <SendToLearn workId={work.id} passage={send.passage} subjects={subjects} onDone={(sub) => setSend({ passage: send.passage, sent: sub })} onCancel={() => setSend(null)} />
            )}
          </div>
        ) : null}
        {mark ? (
          <div className="mx-auto max-w-3xl space-y-3 px-6 pb-4">
            <blockquote className="border-l-2 pl-3 font-serif text-base italic">{mark.passage}</blockquote>
            {mark.kept !== undefined ? (
              <p className="text-sm">
                Kept{mark.kept ? <> in <Link href={`/journal/d/${mark.kept}`} className="underline underline-offset-2">{DOMAINS.find((d) => d.id === mark.kept)!.title}</Link></> : null}.
                <button type="button" className="text-muted-foreground ml-2 text-xs" onClick={() => setMark(null)}>Close</button>
              </p>
            ) : (
              <Mark target={{ workId: work.id, passage: mark.passage }} domain={domain} ideals={ideals} onDone={(d) => setMark({ passage: mark.passage, kept: d })} onCancel={() => setMark(null)} />
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
