// A bundle: one person's journal as one JSON document, the same whichever host it came from. Moving an
// instance between hosts is an export from one and an import into the other. Times and decisions are
// kept as they were; an import goes only into an empty journal.
import type { Entry } from "./journal.ts";
import type { LabelRow, SelfTermRow } from "./store.ts";

export type Labeling = { entry: string; at: string; labels: LabelRow[]; terms: SelfTermRow[] };

export type Bundle = {
  reflect_bundle: 1;
  exported_at: string;
  /** Oldest first, each with its parts and reflections. */
  entries: Entry[];
  /** Oldest first. */
  labelings: Labeling[];
};

export function makeBundle(entries: readonly Entry[], labelings: readonly Labeling[], exportedAt: string): Bundle {
  return {
    reflect_bundle: 1,
    exported_at: exportedAt,
    entries: [...entries].sort((a, b) => a.createdAt.localeCompare(b.createdAt)).map(({ labels: _labels, ...e }) => ({ ...e, reflections: e.reflections ?? [] })),
    labelings: [...labelings].sort((a, b) => a.at.localeCompare(b.at)),
  };
}

export function parseBundle(text: string): Bundle {
  const b = JSON.parse(text) as Partial<Bundle>;
  if (b.reflect_bundle !== 1 || !Array.isArray(b.entries) || !Array.isArray(b.labelings)) throw new Error("not a Reflect bundle");
  const ids = new Set(b.entries.map((e) => e.id));
  for (const e of b.entries) {
    if (typeof e.id !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(e.localDate) || !Array.isArray(e.parts) || e.parts.length === 0) throw new Error(`bundle entry ${String(e.id)} is malformed`);
  }
  for (const l of b.labelings) if (!ids.has(l.entry)) throw new Error(`a labeling names an entry not in the bundle: ${l.entry}`);
  return b as Bundle;
}
