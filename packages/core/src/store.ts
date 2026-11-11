// The store interface: what the app and the CLI ask of wherever a person's records are hosted.
// Two hosts implement it: a folder (a git repository the person owns) and our Postgres.
import type { Entry } from "./journal.ts";
import type { IsoDate } from "./person.ts";

/** One person's journal. Everything is append-only: parts and answers are added, never changed. */
export interface Journal {
  createEntry(text: string, localDate: IsoDate, promptId: string | null): Promise<string>;
  addPart(entryId: string, text: string): Promise<void>;
  closeEntry(entryId: string): Promise<void>;
  /** Newest first, with reflections. */
  listEntries(limit?: number): Promise<Entry[]>;
  /** The person's own words for an entry, parts joined by a blank line, for the reader. */
  entryText(entryId: string): Promise<string | null>;
  startReflection(entryId: string, frame: string): Promise<string>;
  /** The question comes from the frame by its position, never from the caller. */
  keepAnswer(reflectionId: string, seq: number, text: string): Promise<void>;
}

