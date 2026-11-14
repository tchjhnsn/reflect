// The home screen's tiles: one per area of life, several opening the same app (docs/design/REFLECT_UI_2026-11-12.md,
// sections 1 to 3). A tile's tone says its state: in action when something in its app is under way, in thought
// when something is stated, unstated otherwise. MINE's tiles all open their apps (Learn, the journal, and the
// apps in `@reflect/core/life`); OURS, the shared spaces, is not built yet.
import { TILE_KIND } from "@reflect/core/life";

/** Solid glyphs on a 100 by 100 box, drawn for Reflect. */
export const GLYPHS: Record<string, string> = {
  learn: "M10 25 Q30 18 48 26 L48 80 Q30 72 10 78 Z M52 26 Q70 18 90 25 L90 78 Q70 72 52 80 Z",
  work: "M36 28 L36 18 L64 18 L64 28 L86 28 L86 80 L14 80 L14 28 Z M42 28 L58 28 L58 24 L42 24 Z",
  income: "M20 72 h60 v10 h-60 z M20 58 h60 v10 h-60 z M20 44 h60 v10 h-60 z M30 30 h50 v10 h-50 z",
  food: "M8 40 h84 v6 h-84 z M14 50 h72 Q80 80 50 82 Q20 80 14 50 Z",
  water: "M50 12 C50 12 22 46 22 62 A28 28 0 0 0 78 62 C78 46 50 12 50 12 Z",
  shelter: "M50 14 L88 46 L80 46 L80 84 L58 84 L58 62 L42 62 L42 84 L20 84 L20 46 L12 46 Z",
  health: "M50 84 L18 52 A18 18 0 0 1 50 26 A18 18 0 0 1 82 52 Z",
  fitness: "M30 38 L30 30 A20 20 0 0 1 70 30 L70 38 L62 38 L62 30 A12 12 0 0 0 38 30 L38 38 Z M50 40 A24 24 0 1 1 49.99 40 Z",
  transport: "M50 14 A36 36 0 1 1 49.99 14 Z M50 34 A16 16 0 1 0 50.01 34 Z",
  safety: "M50 12 L84 24 L84 48 Q84 74 50 88 Q16 74 16 48 L16 24 Z",
  friends: "M34 22 A10 10 0 1 1 33.99 22 Z M66 22 A10 10 0 1 1 65.99 22 Z M18 82 Q18 46 34 46 Q48 46 48 82 Z M52 82 Q52 46 66 46 Q82 46 82 82 Z",
  standing: "M30 86 h40 v-6 h-40z M36 76 h28 v-44 h-28z M30 28 h40 v-8 h-40z",
  mate: "M50 34 A24 24 0 1 1 49.99 34 Z M50 44 A14 14 0 1 0 50.01 44 Z M42 12 L58 12 L64 20 L50 32 L36 20 Z",
  kin: "M14 50 Q14 80 50 82 Q86 80 86 50 L74 50 Q72 68 50 70 Q28 68 26 50 Z M50 22 A10 10 0 1 1 49.99 22 Z",
  polis: "M50 12 L90 32 L10 32 Z M16 38 h10 v36 h-10z M36 38 h10 v36 h-10z M54 38 h10 v36 h-10z M72 38 h10 v36 h-10z M10 78 h80 v8 h-80z",
  legacy: "M24 12 h52 v8 h-4 Q72 42 54 50 Q72 58 72 80 h4 v8 h-52 v-8 h4 Q28 58 46 50 Q28 42 28 20 h-4 Z",
  ideas: "M50 12 A26 26 0 0 1 64 60 L64 70 L36 70 L36 60 A26 26 0 0 1 50 12 Z M38 74 h24 v6 h-24z M42 84 h16 v4 h-16z",
  journal: "M26 12 h48 a4 4 0 0 1 4 4 v68 a4 4 0 0 1 -4 4 h-48 Z M14 22 h10 v6 h-10z M14 46 h10 v6 h-10z M14 70 h10 v6 h-10z",
  marriage: "M38 30 A20 20 0 1 1 37.99 30 Z M38 38 A12 12 0 1 0 38.01 38 Z M62 30 A20 20 0 1 1 61.99 30 Z M62 38 A12 12 0 1 0 62.01 38 Z",
  household: "M28 10 h44 v80 h-44z M62 48 a4 4 0 1 1 -0.01 0 Z",
  home: "M50 16 L90 48 L80 48 L80 86 L20 86 L20 48 L10 48 Z M66 22 h10 v18 l-10 -8z",
  budget: "M34 12 h32 v8 h-32z M30 24 h40 Q80 30 80 50 L80 80 Q80 88 72 88 L28 88 Q20 88 20 80 L20 50 Q20 30 30 24 Z",
  children: "M36 16 A10 10 0 1 1 35.99 16 Z M20 86 L22 44 Q36 38 50 44 L52 86 Z M70 42 A8 8 0 1 1 69.99 42 Z M58 86 L60 62 Q70 58 80 62 L82 86 Z",
  village: "M10 54 L28 36 L46 54 L46 86 L10 86 Z M54 46 L72 20 L90 46 L90 86 L54 86 Z",
};

export type App = "learn" | "work" | "body" | "home" | "friends" | "mate" | "kin" | "polis" | "legacy" | "journal" | "marriage" | "household" | "children" | "village";
export type Tile = { id: string; label: string; app: App };

export const MINE: readonly Tile[] = [
  { id: "learn", label: "Learn", app: "learn" },
  { id: "work", label: "Work", app: "work" },
  { id: "income", label: "Income", app: "work" },
  { id: "food", label: "Food", app: "home" },
  { id: "water", label: "Water", app: "home" },
  { id: "shelter", label: "Shelter", app: "home" },
  { id: "health", label: "Health", app: "body" },
  { id: "fitness", label: "Fitness", app: "body" },
  { id: "transport", label: "Transport", app: "home" },
  { id: "safety", label: "Safety", app: "home" },
  { id: "friends", label: "Friends", app: "friends" },
  { id: "standing", label: "Standing", app: "polis" },
  { id: "mate", label: "Mate", app: "mate" },
  { id: "kin", label: "Kin", app: "kin" },
  { id: "polis", label: "Polis", app: "polis" },
  { id: "legacy", label: "Legacy", app: "legacy" },
  { id: "ideas", label: "Ideas", app: "journal" },
  { id: "journal", label: "Journal", app: "journal" },
];

export const OURS: readonly Tile[] = [
  { id: "marriage", label: "Marriage", app: "marriage" },
  { id: "household", label: "Household", app: "household" },
  { id: "home", label: "Home", app: "household" },
  { id: "budget", label: "Budget", app: "household" },
  { id: "children", label: "Children", app: "children" },
  { id: "village", label: "Village", app: "village" },
];

/**
 * Where a tile goes: Learn, the journal (Ideas and Journal), or its app in `@reflect/core/life`, at the part it
 * names (Income opens Work's income, Food opens Home's food). OURS's tiles go nowhere yet.
 */
export function hrefOf(t: Tile): string | null {
  if (t.app === "learn") return "/learn";
  if (t.app === "journal") return "/journal";
  const life = TILE_KIND[t.id];
  return life ? `/life/${life.app}${life.kind ? `?k=${life.kind}` : ""}` : null;
}

export type Tone = "action" | "thought" | "unstated";
export type Facts = { learn: { underway: boolean; stated: boolean }; journal: { stated: boolean; open: boolean }; /** Each life tile's tone, from `lifeTone`. */ life?: Record<string, Tone> };

/** A tile's tone from what the person has, never from a guess. */
export function toneOf(t: Tile, f: Facts): Tone {
  if (t.app === "learn") return f.learn.underway ? "action" : f.learn.stated ? "thought" : "unstated";
  if (t.id === "journal") return f.journal.open ? "action" : f.journal.stated ? "thought" : "unstated";
  return f.life?.[t.id] ?? "unstated";
}
