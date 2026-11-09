// Reflecting on an entry: the person picks a frame, then answers fixed, open questions one at a time.
// The decision frame follows the Lean will_order (structure.json). Stance comes from the facet: conscience is
// normative, everything else descriptive (lean/NTS/Frames.lean). No model chooses or writes these questions.
import raw from "./structure.json";

export type Stance = "descriptive" | "normative";
export type Question = { facet: string; stance: Stance; text: string };
export type FrameId = "feeling" | "decision" | "belief" | "identity";
export type Frame = { id: FrameId; root: string; title: string; about: string; questions: readonly Question[] };

const stanceOf = (facet: string): Stance => (facet === "conscience" ? "normative" : "descriptive");

// Generic wording for the decision frame, one per facet in the Lean will_order.
const DECISION: Record<string, string> = {
  voluntary_involuntary: "Was this something you chose, something that happened without your choosing, or something you felt forced into?",
  desire: "How much did you want it, if at all?",
  attachment: "Who, if anyone, were you doing it for or with?",
  imagination: "What else could you have done instead?",
  reason: "What made you pick this over the other options?",
  conscience: "Did you feel you ought to? Why or why not?",
  temporality: "When did you decide, and how long did it take?",
  emotion: "What did you feel as you decided?",
  embodiment: "How was your body when you decided: rested, tired, restless, in pain?",
  intuition: "Did you have a sense, before thinking it through, of what to do?",
  "self-differentiation": "Does this choice feel like you, or like someone else's idea of you?",
  attention: "What were you focused on when you decided?",
  awareness: "Were you aware of deciding at the time, or did you notice only afterwards?",
  logos: "How would you put the decision into words?",
  "aesthetic-sense": "Was there anything about it you found beautiful or appealing in itself?",
  "numinous-sense": "Did it touch anything you hold sacred?",
  will: "How firmly did you hold to the decision once you made it?",
};

type WillStep = { facet: { axis?: string; prime?: string }; stance: Stance };
const willOrder = (raw as { will_order: WillStep[] }).will_order;

function decisionQuestions(): Question[] {
  return willOrder.map((s) => {
    const facet = s.facet.axis ?? s.facet.prime ?? "";
    const text = DECISION[facet];
    if (!text) throw new Error(`no decision question for facet ${facet}`);
    return { facet, stance: s.stance, text };
  });
}

const q = (facet: string, text: string): Question => ({ facet, stance: stanceOf(facet), text });

export const FRAMES: readonly Frame[] = [
  {
    id: "feeling",
    root: "emotion",
    title: "A feeling",
    about: "Why you felt the way you did.",
    questions: [
      q("emotion", "What did you feel, in your own words?"),
      q("embodiment", "Where did you notice it in your body, if anywhere?"),
      q("temporality", "When did it start, and is it still there?"),
      q("attachment", "Who, if anyone, was part of it?"),
      q("attention", "What were you focused on when it came?"),
      q("reason", "What do you think brought it on?"),
      q("desire", "Was there something you wanted, or wanted to avoid?"),
      q("conscience", "Did you feel you ought to feel differently? Why or why not?"),
      q("logos", "If you told a friend about it, what would you say?"),
    ],
  },
  { id: "decision", root: "will", title: "A decision", about: "How and why you chose.", questions: decisionQuestions() },
  {
    id: "belief",
    root: "reason",
    title: "A belief",
    about: "What you think is true here, and why.",
    questions: [
      q("reason", "What do you believe about this?"),
      q("temporality", "When did you come to believe it?"),
      q("attachment", "Who, if anyone, shaped this belief?"),
      q("intuition", "Did you sense it before you could explain it?"),
      q("imagination", "What would it look like if it weren't true?"),
      q("awareness", "How sure are you, and what would change your mind?"),
      q("conscience", "Is this about how things are, or how they ought to be?"),
      q("logos", "How would you put it in one sentence?"),
    ],
  },
  {
    id: "identity",
    root: "self-differentiation",
    title: "Who I am",
    about: "Whether this felt like you.",
    questions: [
      q("self-differentiation", "Did this feel like you? In what way?"),
      q("embodiment", "How were you in your body at the time?"),
      q("attachment", "Who were you with, or thinking of?"),
      q("temporality", "Would you have done or felt the same a few years ago?"),
      q("desire", "What did you want for yourself here?"),
      q("conscience", "Is there a way you feel you ought to be here? Where does that come from?"),
      q("aesthetic-sense", "Was there anything in it you found beautiful or fitting?"),
      q("numinous-sense", "Did it touch anything you hold sacred?"),
    ],
  },
];

export function frameById(id: string): Frame | null {
  return FRAMES.find((f) => f.id === id) ?? null;
}

const LEADING = /\b(weren'?t|didn'?t|don'?t|wasn'?t|isn'?t) you\b|\byou (must|should|will|would|probably)\b|\bshould you\b/i;

/** The same rule as thrivesight_core/nts/investigate.py: open questions only, never leading or prescriptive. */
export function isOpenQuestion(text: string): boolean {
  return text.trim().endsWith("?") && !LEADING.test(text);
}

export type Answer = { seq: number; facet: string; stance: Stance; question: string; text: string | null };

export type ReflectState = { frame: FrameId; step: number; answers: Answer[] };

/** The next question, or null when every question has been answered or skipped. */
export function current(s: ReflectState): Question | null {
  return frameById(s.frame)?.questions[s.step] ?? null;
}

export function answer(s: ReflectState, text: string | null): ReflectState {
  const question = current(s);
  if (!question) return s;
  const words = text?.trim() || null;
  const a: Answer = { seq: s.step, facet: question.facet, stance: question.stance, question: question.text, text: words };
  return { ...s, step: s.step + 1, answers: [...s.answers, a] };
}

/** Descriptive and normative answers, kept apart. Skipped questions are left out. */
export function split(answers: readonly Answer[]): { descriptive: Answer[]; normative: Answer[] } {
  const given = answers.filter((a) => a.text !== null);
  return { descriptive: given.filter((a) => a.stance === "descriptive"), normative: given.filter((a) => a.stance === "normative") };
}
