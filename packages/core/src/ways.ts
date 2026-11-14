/**
 * The ways a person learns, after Aristotle, which arrange Learn's shelves; the medium (book, film, lecture, game)
 * is a property of each item, not its shelf (Torian, 2026-11-14).
 *
 * - teaching (didaskalia): one who knows the causes shows them (Posterior Analytics I.1; Metaphysics I.1).
 * - dialectic: reputable opinions tested by question and answer (Topics I.1-2).
 * - imitation (mimesis): learning by recognising "this is that" in an image or story (Poetics 4, 9).
 * - practice: skill and character formed by doing (Nicomachean Ethics II.1).
 *
 * Play is rest for the sake of serious activity (NE X.6) and music serves play, character and leisure (Politics
 * VIII.5), so a game, a piece of music or a work of art counts as learning only once the person says what it
 * taught them; until then it is kept under rest and leisure.
 */

export const WAYS = [
  { id: "teaching", label: "Teaching", about: "Someone who knows shows the causes: lectures, courses, books that teach." },
  { id: "dialectic", label: "Dialectic", about: "Opinions tested by question and answer: debates, discussions, dialogues." },
  { id: "imitation", label: "Imitation", about: "Learning by recognising life in a likeness: stories, drama, film, art, music." },
  { id: "practice", label: "Practice", about: "Formed by doing: games, crafts and exercises that built a skill or a habit." },
] as const;
export type Way = (typeof WAYS)[number]["id"];
export const isWay = (w: unknown): w is Way => typeof w === "string" && WAYS.some((x) => x.id === w);

/** Kept, but not yet learning: play and leisure the person has not said taught them anything. */
export const LEISURE = { id: "leisure", label: "Rest and leisure", about: "Games, music and art kept for rest and enjoyment, until you say what one taught you." } as const;

export const MEDIA = [
  { id: "lecture", label: "Lecture", way: "teaching", done: "Watched", doing: "Watching", lesson: false },
  { id: "course", label: "Course", way: "teaching", done: "Finished", doing: "Taking", lesson: false },
  { id: "article", label: "Article", way: "teaching", done: "Read", doing: "Reading", lesson: false },
  { id: "debate", label: "Debate", way: "dialectic", done: "Watched", doing: "Watching", lesson: false },
  { id: "discussion", label: "Discussion", way: "dialectic", done: "Heard", doing: "Listening", lesson: false },
  { id: "podcast", label: "Podcast", way: "dialectic", done: "Heard", doing: "Listening", lesson: false },
  { id: "film", label: "Film", way: "imitation", done: "Watched", doing: "Watching", lesson: false },
  { id: "series", label: "Series", way: "imitation", done: "Watched", doing: "Watching", lesson: false },
  { id: "theatre", label: "Theatre", way: "imitation", done: "Seen", doing: "Seeing", lesson: false },
  { id: "music", label: "Music", way: "imitation", done: "Heard", doing: "Listening", lesson: true },
  { id: "art", label: "Art", way: "imitation", done: "Seen", doing: "Looking", lesson: true },
  { id: "game", label: "Game", way: "practice", done: "Played", doing: "Playing", lesson: true },
] as const satisfies readonly { id: string; label: string; way: Way; done: string; doing: string; lesson: boolean }[];
export type Medium = (typeof MEDIA)[number]["id"];
export const isMedium = (m: unknown): m is Medium => typeof m === "string" && MEDIA.some((x) => x.id === m);
export const mediumOf = (m: Medium) => MEDIA.find((x) => x.id === m)!;

export const ITEM_STATUS = ["want", "doing", "done"] as const;
export type ItemStatus = (typeof ITEM_STATUS)[number];

/**
 * Where a book sits by default: written dialogues (a catalog subject naming dialogues, as Plato's do) by
 * dialectic; stories, poetry and drama by imitation (Thema F fiction, DC poetry, DD drama, and the children's
 * fiction sections); every other book by teaching. The person can move it.
 */
export function bookWay(category: string | null | undefined, subjects: readonly string[] = []): Way {
  if (subjects.some((s) => /\bdialogues?\b/i.test(s))) return "dialectic";
  const c = (category ?? "").toUpperCase();
  return c.startsWith("F") || c === "DC" || c === "DD" || c === "YF" || c === "YB" ? "imitation" : "teaching";
}

/**
 * Where an item is kept: the way the person chose, else its medium's; a game, music or art without a lesson is
 * kept under rest and leisure whatever its way.
 */
export function placeOf(item: { medium: Medium; way?: Way | null; lesson?: string | null }): Way | "leisure" {
  const m = mediumOf(item.medium);
  if (m.lesson && !item.lesson?.trim()) return "leisure";
  return item.way ?? m.way;
}
