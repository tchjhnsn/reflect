// Prompt packs the person opts into. Fixed text, no model. Every prompt asks; none assumes an answer, prescribes or predicts.

export type Prompt = { id: string; text: string };
export type Pack = { id: string; title: string; about: string; element: string; prompts: readonly Prompt[] };

const pack = (id: string, title: string, about: string, element: string, texts: string[]): Pack => ({
  id,
  title,
  about,
  element,
  prompts: texts.map((text, i) => ({ id: `${id}.${i + 1}`, text })),
});

export const PACKS: readonly Pack[] = [
  pack("retrospection", "Looking back", "Questions about what has already happened.", "Temporality", [
    "What happened today that you want to remember?",
    "What is something from this week you keep coming back to?",
    "What was different about today compared with yesterday?",
    "When did time pass quickly today, and when slowly?",
    "What is a moment from a year ago you still think about?",
    "What did you notice today that you usually miss?",
  ]),
  pack("growth", "Changing", "Questions about how you are changing, in your own terms.", "Self-differentiation", [
    "What is something you did today that felt like you?",
    "What is something you did today that didn't feel like you?",
    "What are you learning right now, and how is it going?",
    "What would you like to be different a month from now, if anything?",
    "What did you try today that you hadn't tried before?",
    "What do you understand now that you didn't a year ago?",
  ]),
  pack("decisions", "Decisions", "Questions about choices you made or are making.", "Will", [
    "What did you decide today, and how did you decide it?",
    "What choice are you weighing right now? What are the options?",
    "Was there something today you felt you had to do? Where did that come from?",
    "What did you want today that you didn't get, or did?",
    "Is there a decision you made that you would make differently? Why?",
    "Who, if anyone, did you have in mind when you chose?",
  ]),
  pack("body", "Body check-in", "Questions about how your body is.", "Embodiment", [
    "How is your body right now: rested, tired, restless, in pain?",
    "How did you sleep, and how did it shape your day?",
    "Where in your body did you notice something today?",
    "What did you eat or drink today that you noticed?",
    "When did you move today, and how did it feel?",
    "What does your body want right now, if anything?",
  ]),
  pack("meaning", "Gratitude and meaning", "Questions about what you find beautiful, good or sacred.", "Aesthetic and Numinous", [
    "What is something you are grateful for today?",
    "What did you see, hear or taste today that you found beautiful?",
    "What, if anything, felt meaningful today?",
    "Who are you glad is in your life, and why?",
    "Is there anything you hold sacred that came up today?",
    "What made you laugh or smile today?",
  ]),
];

const BY_ID = new Map(PACKS.flatMap((p) => p.prompts.map((q) => [q.id, q] as const)));

export function promptById(id: string | null): Prompt | null {
  return id === null ? null : (BY_ID.get(id) ?? null);
}

export function isPackId(id: string): boolean {
  return PACKS.some((p) => p.id === id);
}

/** The prompts of the chosen packs, in pack order. */
export function pool(chosen: readonly string[]): Prompt[] {
  return PACKS.filter((p) => chosen.includes(p.id)).flatMap((p) => [...p.prompts]);
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/**
 * Draw a prompt from the chosen packs only. Deterministic: the same day and turn give the same prompt.
 * Skips prompts already used, until all have been used; returns null when no pack is chosen.
 */
export function drawPrompt(chosen: readonly string[], used: readonly string[], day: string, turn: number): Prompt | null {
  const all = pool(chosen);
  if (all.length === 0) return null;
  const fresh = all.filter((p) => !used.includes(p.id));
  const from = fresh.length > 0 ? fresh : all;
  return from[hash(`${day}:${turn}`) % from.length] ?? null;
}
