// The store interface: what the app and the CLI ask of wherever a person's records are hosted.
// Two hosts implement it: a folder (a git repository the person owns) and our Postgres.
import type { Entry } from "./journal.ts";
import type { IsoDate } from "./person.ts";

export type LabelRow = { element: string; weight: number | null; decision: "confirmed" | "rejected" | "added" | "absent" };
export type SelfTermRow = {
  quote: string;
  term: string;
  whose: "self" | "other";
  modelBasis: string | null;
  basis: string | null;
  decision: "confirmed" | "corrected" | "rejected";
};

/** One person's journal. Everything is append-only: parts, answers and labelings are added, never changed. */
export interface Journal {
  createEntry(text: string, localDate: IsoDate, promptId: string | null): Promise<string>;
  addPart(entryId: string, text: string): Promise<void>;
  closeEntry(entryId: string): Promise<void>;
  /** Newest first, with reflections and the elements of the latest labeling. */
  listEntries(limit?: number): Promise<Entry[]>;
  /** The person's own words for an entry, parts joined by a blank line, for the reader. */
  entryText(entryId: string): Promise<string | null>;
  startReflection(entryId: string, frame: string): Promise<string>;
  /** The question comes from the frame by its position, never from the caller. */
  keepAnswer(reflectionId: string, seq: number, text: string): Promise<void>;
  keepLabels(entryId: string, labels: LabelRow[], terms: SelfTermRow[]): Promise<void>;
}

/** The elements a labeling keeps: those the person confirmed or added. */
export const kept = (labels: readonly LabelRow[]): string[] => labels.filter((l) => l.decision === "confirmed" || l.decision === "added").map((l) => l.element);
