import raw from "./structure.json";

export type Destination =
  | { store: "profile"; key: string }
  | { store: "relationships"; kind: "closest_friend" };

export type OnboardingQuestion = { id: string; destination: Destination };

function fail(path: string): never {
  throw new Error(`structure.json: missing or malformed ${path}`);
}

function parseDestination(value: unknown, path: string): Destination {
  if (typeof value !== "object" || value === null) fail(path);
  const d = value as Record<string, unknown>;
  if (d.store === "profile" && typeof d.key === "string") return { store: "profile", key: d.key };
  if (d.store === "relationships" && d.kind === "closest_friend") return { store: "relationships", kind: "closest_friend" };
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
