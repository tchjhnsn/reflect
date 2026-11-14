// The journal as a library of the person's works (Torian, 2026-11-12 18:39Z and 18:48Z): not a page of the
// day and not a form, but a writing editor. A work is anything the person writes: an entry, an essay, a
// letter, a plan. Each is kept as versions, never overwritten. The prompt library is how a work can start:
// a template for a kind of thinking, which the person writes over freely. Kinds name forms after the Library
// of Congress Genre/Form Terms (LCGFT) where one fits, as decided for the library of forms (11-11 21:30Z).

export type WorkKind = "entry" | "essay" | "letter" | "plan" | "examination" | "exercise" | "decision" | "memoir";

/** How each kind reads, and the LCGFT term it follows where there is one. */
export const KINDS: Record<WorkKind, { label: string; lcgft: string | null }> = {
  entry: { label: "Journal entry", lcgft: "Diaries" },
  essay: { label: "Essay", lcgft: "Essays" },
  letter: { label: "Letter", lcgft: "Personal correspondence" },
  plan: { label: "Plan", lcgft: null },
  examination: { label: "Examination", lcgft: null },
  exercise: { label: "Exercise", lcgft: null },
  decision: { label: "Decision", lcgft: null },
  memoir: { label: "Memoir", lcgft: "Autobiographies" },
};

export type Template = {
  id: string;
  name: string;
  kind: WorkKind;
  /** One line on when to use it. */
  about: string;
  /** Where it comes from, when it follows a known practice. */
  source?: string;
  /** The starting text, as Markdown. Headings are the prompts; the person writes under them or deletes them. */
  body: string;
};

/** The prompt library: templates for kinds of thinking. Blank first. */
export const TEMPLATES: Template[] = [
  { id: "blank", name: "Blank page", kind: "entry", about: "Just write.", body: "" },
  {
    id: "object-writing", name: "Object writing", kind: "exercise",
    about: "Pick one thing and write about it for ten minutes, through every sense.",
    source: "Pat Pattison, Writing Better Lyrics (object writing)",
    body: "## The object\n\n\n\n## What I see, hear, smell, taste, touch\n\n\n\n## What it reminds me of\n\n",
  },
  {
    id: "examine-belief", name: "Examine a belief", kind: "examination",
    about: "Say what you hold, ground it, and set it against the strongest case.",
    source: "Naess (precization), Rawls (reflective equilibrium)",
    body: "## What I hold\n\n\n\n## Said precisely\n\n\n\n## Why I hold it\n\n\n\n## The strongest case against it\n\n\n\n## What others told me\n\n\n\n## Where I stand now\n\n",
  },
  {
    id: "essay", name: "Essay", kind: "essay",
    about: "Make a case: a thesis, the argument, and what follows.",
    body: "## Thesis\n\n\n\n## Argument\n\n\n\n## Objections\n\n\n\n## Conclusion\n\n",
  },
  {
    id: "decision", name: "A decision", kind: "decision",
    about: "Lay out a choice before you make it.",
    source: "Aristotle, Nicomachean Ethics III.3 (deliberation is about what is in our power)",
    body: "## The choice\n\n\n\n## What is in my power\n\n\n\n## The options\n\n\n\n## What I want it to serve\n\n\n\n## What I choose, and why\n\n",
  },
  {
    id: "plan", name: "Plan", kind: "plan",
    about: "Decide how to carry something out.",
    body: "## What I mean to do\n\n\n\n## Why\n\n\n\n## Steps\n\n- \n\n## When\n\n",
  },
  {
    id: "why-learn", name: "Why I am learning this", kind: "entry",
    about: "Before you take up a course or a subject: what is causing you to learn it, and for the sake of what.",
    source: "Aristotle, Rhetoric I.10 (the seven causes of action) and Physics II.3 (the moving cause and the end)",
    body: "## What is causing me to learn this\n\n\n\n## For the sake of what\n\n",
  },
  { id: "letter", name: "Letter", kind: "letter", about: "Write to someone, sent or not.", body: "Dear ,\n\n\n\n" },
  {
    id: "legacy", name: "What I want the world to know", kind: "memoir",
    about: "For the people who come after you.",
    source: "Chochinov, Dignity Therapy question protocol",
    body: "## What I most want my family to know about me\n\n\n\n## The things I am proudest of\n\n\n\n## What I have learned about life\n\n",
  },
];

export const templateById = (id: string): Template | undefined => TEMPLATES.find((t) => t.id === id);

/** A work's title as shown: the title it was given, else its first line of text, else its kind. */
export function workTitle(w: { title: string; body: string; kind: WorkKind }): string {
  if (w.title.trim()) return w.title.trim();
  const line = w.body.split("\n").map((l) => l.replace(/^#+\s*|[*_>`-]/g, "").trim()).find(Boolean);
  return line ? line.slice(0, 80) : `Untitled ${KINDS[w.kind].label.toLowerCase()}`;
}

/** The text a version keeps: line ends evened out, no trailing space, at most 200,000 characters. */
export function cleanWork(title: unknown, body: unknown): { title: string; body: string } {
  const t = typeof title === "string" ? title.replace(/\s+/g, " ").trim().slice(0, 300) : "";
  const b = typeof body === "string" ? body.replace(/\r\n?/g, "\n").replace(/[ \t]+$/gm, "").trimEnd().slice(0, 200_000) : "";
  return { title: t, body: b };
}

export const isWorkKind = (k: unknown): k is WorkKind => typeof k === "string" && k in KINDS;

/**
 * A work from an imported file (Markdown or plain text). The title is a leading `# ` heading, which leaves
 * the body, else the file's name without its extension. YAML front matter is kept as written.
 */
export function workFromFile(name: string, text: string): { title: string; body: string } {
  const body = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n");
  const m = /^(?:\s*\n)*# +(.+)\n?/.exec(body);
  const base = name.replace(/^.*[\\/]/, "").replace(/\.(md|markdown|txt|text)$/i, "").replace(/[-_]+/g, " ");
  return m ? cleanWork(m[1], body.slice(m[0].length).replace(/^\n+/, "")) : cleanWork(base, body);
}

/** Files the journal imports: Markdown and plain text, up to 1 MB each. */
export const IMPORTABLE = /\.(md|markdown|txt|text)$/i;
export const MAX_IMPORT_BYTES = 1_000_000;
