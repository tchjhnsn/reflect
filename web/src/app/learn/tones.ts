// How a book or subject looks on a shelf: dark is known, grey is learning, light is registered (the canvas
// legend). Taken only from what the person said: a book's status, a subject's grasp.
import type { BookStatus, Grasp } from "@reflect/core/learning";

export type ShelfTone = "known" | "learning" | "registered";

export const bookTone = (s: BookStatus): ShelfTone => (s === "read" ? "known" : s === "reading" ? "learning" : "registered");

export function subjectTone(grasp: Grasp | null | undefined, onBooks: readonly BookStatus[]): ShelfTone {
  if (grasp === "teach" || grasp === "explain" || grasp === "gist") return "known";
  return onBooks.some((s) => s === "reading" || s === "read") ? "learning" : "registered";
}
