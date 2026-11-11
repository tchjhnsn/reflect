// Where a person's records sit in an instance folder. A vault (the default since 2026-11-10) is the
// folder itself, opened as is in Obsidian or any editor: journal/, reflections/, library/, one folder
// per venture of a lifetime, telos/, intents/, concepts/. The older "people" layout kept the same
// records under people/<id>/ with entries/ for the journal; instances made that way still work.
import { join } from "node:path";

export type Layout = "vault" | "people";

/** The ventures of a lifetime, each a folder under ventures/: entering a household, building work, community, friendship. */
export const VENTURES_OF_A_LIFETIME = ["household", "work", "community", "friendship"] as const;

export type RecordPaths = {
  /** The person's root: "" in a vault, people/<id> otherwise. */
  person: string;
  journal: string;
  reflections: string;
  concepts: string;
  grants: string;
  /** Custody cards for official documents, one per document type. */
  documents: string;
  /** Services that hold records for the person (a secure vault, a school's credential service). */
  connections: string;
  /** Folders whose files are the person's records, checked against memory. */
  recordDirs: string[];
  /** Single files that are records. */
  recordFiles: string[];
};

export function recordPaths(layout: Layout, personId: string): RecordPaths {
  const person = layout === "vault" ? "" : join("people", personId);
  const at = (p: string) => (person ? join(person, p) : p);
  const journal = at(layout === "vault" ? "journal" : "entries");
  const dirs = layout === "vault" ? ["library", "ventures", "telos", "intents"] : ["intents", "telos"];
  return {
    person,
    journal,
    reflections: at("reflections"),
    concepts: at("concepts"),
    grants: at("grants"),
    documents: at("documents"),
    connections: at("connections"),
    recordDirs: layout === "vault" ? [journal, at("reflections"), at("concepts"), at("grants"), at("documents"), at("connections"), ...dirs.map(at)] : [person],
    recordFiles: [at("person.json"), "reflect.json"],
  };
}
