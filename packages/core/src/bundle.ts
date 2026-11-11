// A bundle: one person's journal as one JSON document, the same whichever host it came from. Moving an
// instance between hosts is an export from one and an import into the other. Times are kept as they were;
// an import goes only into an empty journal. Bundles exported before 2026-11-11 also carry element
// labelings; those belong to the soul theory, which left Reflect, and an import passes over them.
import type { Entry } from "./journal.ts";

export type Bundle = {
  reflect_bundle: 1;
  exported_at: string;
  /** Oldest first, each with its parts and reflections. */
  entries: Entry[];
};

export function makeBundle(entries: readonly Entry[], exportedAt: string): Bundle {
  return {
    reflect_bundle: 1,
    exported_at: exportedAt,
    entries: [...entries].sort((a, b) => a.createdAt.localeCompare(b.createdAt)).map((e) => ({ ...e, reflections: e.reflections ?? [] })),
  };
}

export function parseBundle(text: string): Bundle {
  const b = JSON.parse(text) as Partial<Bundle>;
  if (b.reflect_bundle !== 1 || !Array.isArray(b.entries)) throw new Error("not a Reflect bundle");
  for (const e of b.entries) {
    if (typeof e.id !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(e.localDate) || !Array.isArray(e.parts) || e.parts.length === 0) throw new Error(`bundle entry ${String(e.id)} is malformed`);
  }
  return { reflect_bundle: 1, exported_at: String(b.exported_at ?? ""), entries: b.entries };
}
