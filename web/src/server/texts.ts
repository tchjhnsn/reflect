import { findGutenberg, gutenbergBook, type Chapter } from "@reflect/core/gutenberg";
import { standardBook } from "@reflect/core/standardebooks";
import type { WorkAccess } from "@reflect/core/openlibrary";

/** Where a public-domain book's text comes from: Standard Ebooks (proofread and typeset) or Project Gutenberg. */
export type TextSource = { kind: "standard"; id: string } | { kind: "gutenberg"; id: string };
export type BookText = { source: TextSource; chapters: Chapter[]; credit: { name: string; href: string } };

const cached = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 } });

/**
 * Where to read a book's text in Reflect: its Standard Ebooks edition when the catalog links one, else its
 * Gutenberg text linked from the catalog, else the book found in Gutenberg's own catalog by title and author.
 */
export async function textSource(book: { title: string; authors: readonly string[] }, access: WorkAccess | null): Promise<TextSource | null> {
  if (access?.standardEbooks[0]) return { kind: "standard", id: access.standardEbooks[0] };
  if (access?.gutenberg[0]) return { kind: "gutenberg", id: access.gutenberg[0] };
  const id = await findGutenberg(book, cached).catch(() => null);
  return id ? { kind: "gutenberg", id } : null;
}

/** A book's whole text as chapters, with whom it comes from; a Standard Ebooks edition that fails falls back to Gutenberg's text when there is one. */
export async function readText(source: TextSource, fallback?: TextSource | null): Promise<BookText | null> {
  if (source.kind === "standard") {
    const b = await standardBook(source.id, cached).catch(() => null);
    if (b) return { source, chapters: b.chapters, credit: { name: "Standard Ebooks", href: `https://standardebooks.org/ebooks/${source.id}` } };
    return fallback && fallback.kind !== "standard" ? readText(fallback) : null;
  }
  const b = await gutenbergBook(source.id, cached).catch(() => null);
  return b ? { source, chapters: b.chapters, credit: { name: "Project Gutenberg", href: `https://www.gutenberg.org/ebooks/${source.id}` } } : null;
}
