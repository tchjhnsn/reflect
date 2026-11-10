import raw from "./structure.json";

export type Destination =
  | { store: "profile"; key: string }
  | { store: "relationships"; kind: "closest_friend" }
  | { store: "holds"; subject: string };

export type OnboardingQuestion = { id: string; destination: Destination };

function fail(path: string): never {
  throw new Error(`structure.json: missing or malformed ${path}`);
}

function parseDestination(value: unknown, path: string): Destination {
  if (typeof value !== "object" || value === null) fail(path);
  const d = value as Record<string, unknown>;
  if (d.store === "profile" && typeof d.key === "string") return { store: "profile", key: d.key };
  if (d.store === "relationships" && d.kind === "closest_friend") return { store: "relationships", kind: "closest_friend" };
  if (d.store === "holds" && typeof d.subject === "string") return { store: "holds", subject: d.subject };
  fail(path);
}

export function parseOnboarding(json: unknown): readonly OnboardingQuestion[] {
  const list = (json as { onboarding?: unknown })?.onboarding;
  if (!Array.isArray(list) || list.length === 0) fail("onboarding");
  return list.map((q: unknown, i) => {
    const item = q as Record<string, unknown>;
    if (typeof item?.id !== "string") fail(`onboarding[${i}].id`);
    return { id: item.id, destination: parseDestination(item.destination, `onboarding[${i}].destination`) };
  });
}

export const ONBOARDING: readonly OnboardingQuestion[] = parseOnboarding(raw);

export type HoldId = "desire" | "will" | "duty" | "sacred" | "not_for_me" | "unsure";
export type HoldDef = { id: HoldId; stance: "descriptive" | "normative"; prime: string | null };

export function parseHolds(json: unknown): readonly HoldDef[] {
  const list = (json as { holds?: unknown })?.holds;
  if (!Array.isArray(list) || list.length === 0) fail("holds");
  return list.map((h: unknown, i) => {
    const item = h as { id?: unknown; stance?: unknown; facet?: { prime?: unknown } | null };
    if (typeof item.id !== "string") fail(`holds[${i}].id`);
    if (item.stance !== "descriptive" && item.stance !== "normative") fail(`holds[${i}].stance`);
    const prime = item.facet && typeof item.facet.prime === "string" ? item.facet.prime : null;
    return { id: item.id as HoldId, stance: item.stance, prime };
  });
}

/** How a person can hold partnership or children, from lean/NTS/Onboarding.lean. Duty is normative. */
export const HOLDS: readonly HoldDef[] = parseHolds(raw);
