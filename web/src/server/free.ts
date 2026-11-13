import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { findFree, freeIndex, type FreeIndex } from "@reflect/core/free";
import { knowFree, type Free } from "@reflect/core/openlibrary";

// Whether a book can be read or heard in Reflect: matched against Project Gutenberg's English texts and LibriVox's
// English recordings (src/data, refreshed by scripts/free-catalogs.ts). Only books old enough to be in the public
// domain are matched, so a modern book with a classic's title is not marked. Importing this module turns the marks
// on for every catalog entry the server makes.

const cache = new Map<string, FreeIndex>();
function catalog(name: "gutenberg" | "librivox"): FreeIndex {
  let index = cache.get(name);
  if (!index) {
    const file = [join(process.cwd(), "src/data", `${name}.txt`), join(process.cwd(), "web/src/data", `${name}.txt`)].find((f) => existsSync(f));
    index = freeIndex(file ? readFileSync(file, "utf8") : "");
    cache.set(name, index);
  }
  return index;
}

/** A book's Gutenberg number from the catalog file, or null. */
export const gutenbergId = (book: { title: string; authors?: readonly string[] }): string | null => findFree(catalog("gutenberg"), book);
/** A book's LibriVox id from the catalog file, or null. */
export const librivoxId = (book: { title: string; authors?: readonly string[] }): string | null => findFree(catalog("librivox"), book);

/** Old enough to be free: first published before 1930, or free to read in the open library. */
const old = (b: { year?: number; ebook?: string }) => b.ebook === "public" || (b.year !== undefined && b.year < 1930);

knowFree((b): Free | null => {
  const read = b.gutenberg || b.standard || (old(b) && gutenbergId(b) !== null);
  const listen = b.librivox || (old(b) && librivoxId(b) !== null);
  return read || listen ? { ...(read ? { read: true as const } : {}), ...(listen ? { listen: true as const } : {}) } : null;
});
