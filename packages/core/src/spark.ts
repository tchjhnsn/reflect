// Spark: object writing. Take one concrete noun and write about it for a set time, or in one sentence,
// through the senses (after Pat Pattison, "Writing Better Lyrics", 2nd ed., 2009: sight, hearing, touch,
// taste, smell, the body inside, and movement). Each spark names its noun as a concept, so the notes on
// one noun gather in one place and the person's graph starts from words they chose.

export const SPARK_FORMS = ["1m", "5m", "10m", "15m", "sentence"] as const;
export type SparkForm = (typeof SPARK_FORMS)[number];

/** Concrete nouns to start from when the person has none. Objects you can hold or stand in, not ideas. */
export const SPARK_NOUNS = [
  "kitchen table", "front door", "river", "bicycle", "coffee cup", "window", "key", "shoes", "bridge", "garden",
  "notebook", "bus stop", "rain", "photograph", "pillow", "street lamp", "bread", "hammer", "staircase", "letter",
  "phone", "jacket", "porch", "candle", "map", "wallet", "fence", "radio", "mirror", "suitcase",
  "church pew", "basketball court", "classroom", "car seat", "workbench", "hospital bed", "grocery list", "ladder", "ring", "desk",
] as const;

export function isSparkForm(s: string): s is SparkForm {
  return (SPARK_FORMS as readonly string[]).includes(s);
}

export function pickNoun(rand: () => number = Math.random): string {
  return SPARK_NOUNS[Math.floor(rand() * SPARK_NOUNS.length) % SPARK_NOUNS.length]!;
}

/** A noun as typed, trimmed and made safe as a file name (Obsidian's forbidden characters removed). */
export function cleanNoun(noun: string): string {
  const n = noun.replace(/[\\/:*?"<>|#^[\]\n\r\t]/g, " ").replace(/\s+/g, " ").trim().slice(0, 60);
  if (!n) throw new Error("a spark needs a noun");
  return n;
}

export function sparkPrompt(noun: string, form: SparkForm): string {
  const how = form === "sentence" ? "Write one sentence" : `Write for ${form.replace("m", "")} minute${form === "1m" ? "" : "s"}, without stopping,`;
  return `${how} about ${noun}. Stay with the senses: what you see, hear, touch, taste and smell, what you feel in your body, and how things move.`;
}

/** The front matter a spark entry carries beside the entry's own, as Obsidian properties. */
export function sparkHead(noun: string, form: SparkForm): string[] {
  return [`spark: ${form}`, `concept: "[[${noun}]]"`];
}

/** A concept note: the first time a noun is sparked. Obsidian's backlinks then list every entry on it. */
export function conceptNote(noun: string, at: string): string {
  return `---\nkind: entity\ncreated_at: ${at}\n---\n\n# ${noun}\n`;
}
