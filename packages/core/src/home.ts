import { appOf, MET, PROVIDER, type RecordFacts } from "./life";

/** Needs met by someone other than the person: what Work does not have to earn for. */
export const COVERED_BY = new Set(["kin", "polis", "gifts", "nature"]);

export type NeedView = {
  id: string;
  label: string;
  stated: boolean;
  how: string | null;
  by: string | null;
  byId: string | null;
  covered: boolean;
  cost: number | null;
  done: number;
  of: number;
  missing: string[];
};

const labelOf = (l: readonly { id: string; label: string }[], v: unknown) => l.find((o) => o.id === v)?.label ?? null;

/**
 * Each of Home's needs as the person stated it: how it is met and by whom, what it costs a month, and which parts
 * of its standard are in place and which are not. A need never stated has every part missing.
 */
export function homeNeeds(records: readonly RecordFacts[]): NeedView[] {
  return appOf("home").kinds.filter((k) => k.one).map((k) => {
    const r = records.find((x) => x.kind === k.id);
    const b = r?.body ?? {};
    const checks = k.fields.filter((f) => f.kind === "check");
    const byId = typeof b.by === "string" ? b.by : null;
    return {
      id: k.id, label: k.label, stated: Boolean(r),
      how: labelOf(MET, b.how), by: labelOf(PROVIDER, b.by), byId,
      covered: byId !== null && COVERED_BY.has(byId),
      cost: typeof b.cost === "number" ? b.cost : null,
      done: checks.filter((f) => b[f.key] === true).length, of: checks.length,
      missing: checks.filter((f) => b[f.key] !== true).map((f) => f.label),
    };
  });
}

/** What the person pays for Home a month, and what others cover; null when no cost is given at all. */
export function homeCost(needs: readonly NeedView[]): { pay: number | null; covered: NeedView[] } {
  const paid = needs.filter((n) => !n.covered && n.cost !== null);
  return { pay: paid.length ? paid.reduce((s, n) => s + n.cost!, 0) : null, covered: needs.filter((n) => n.covered) };
}
