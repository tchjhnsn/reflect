// Mirrors lean/NTS/Person.lean and thrivesight_core/nts/person.py.

export type Brand<T, B extends string> = T & { readonly __brand: B };
export type IsoDate = Brand<string, "IsoDate">;

export type Source =
  | { kind: "stated" }
  | { kind: "fromJournal"; entry: string; quote: string; grant: string };

export const STATED: Source = { kind: "stated" };

export type Fact = {
  key: string;
  value: string;
  validFrom: IsoDate;
  validTo: IsoDate | null;
  recordedAt: IsoDate;
  source: Source;
  status: "proposed" | "confirmed";
};

export type Relationship = {
  name: string;
  kind: "closest_friend";
  validFrom: IsoDate;
  validTo: IsoDate | null;
  recordedAt: IsoDate;
  source: { kind: "stated" };
};

export type PersonState = {
  facts: readonly Fact[];
  relationships: readonly Relationship[];
};

export const EMPTY_PERSON: PersonState = { facts: [], relationships: [] };

export function isoDate(text: string): IsoDate | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) return null;
  const d = new Date(`${text}T00:00:00Z`);
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === text ? (text as IsoDate) : null;
}

/** A stated fact closes the open value for its key; nothing is overwritten. */
export function stateFact(p: PersonState, key: string, value: string, at: IsoDate): PersonState {
  const facts = p.facts.map((f) =>
    f.key === key && f.status === "confirmed" && f.validTo === null ? { ...f, validTo: at } : f,
  );
  const fact: Fact = { key, value, validFrom: at, validTo: null, recordedAt: at, source: STATED, status: "confirmed" };
  return { ...p, facts: [...facts, fact] };
}

export function stateRelationship(p: PersonState, name: string, kind: Relationship["kind"], at: IsoDate): PersonState {
  const relationships = p.relationships.map((r) => (r.kind === kind && r.validTo === null ? { ...r, validTo: at } : r));
  const rel: Relationship = { name, kind, validFrom: at, validTo: null, recordedAt: at, source: { kind: "stated" } };
  return { ...p, relationships: [...relationships, rel] };
}

export function currentFact(p: PersonState, key: string): string | null {
  const live = p.facts.filter((f) => f.key === key && f.status === "confirmed" && f.validTo === null);
  return live.at(-1)?.value ?? null;
}

export function currentRelationship(p: PersonState, kind: Relationship["kind"]): string | null {
  return p.relationships.filter((r) => r.kind === kind && r.validTo === null).at(-1)?.name ?? null;
}
