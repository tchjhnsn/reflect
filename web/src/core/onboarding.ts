// Mirrors lean/NTS/Onboarding.lean and thrivesight_core/nts/onboarding.py. No model runs.
import { ONBOARDING, type OnboardingQuestion } from "./structure";
import { isoDate, stateFact, statePuzzle, stateRelationship, type IsoDate, type PersonState } from "./person";

export type OnboardingState = {
  step: number;
  answers: Readonly<Record<string, string>>;
  error: { question: string; message: string } | null;
  /** True while the person edits one answer from the review: the next answer or skip returns there. */
  editing: boolean;
};

export type OnboardingEvent =
  | { type: "answer"; text: string; today: IsoDate }
  | { type: "skip" }
  | { type: "back" }
  | { type: "edit"; step: number };

export const START: OnboardingState = { step: 0, answers: {}, error: null, editing: false };

export function currentQuestion(s: OnboardingState, order: readonly OnboardingQuestion[] = ONBOARDING) {
  return order[s.step] ?? null;
}

export function isDone(s: OnboardingState, order: readonly OnboardingQuestion[] = ONBOARDING): boolean {
  return s.step >= order.length;
}

function check(question: string, text: string, today: IsoDate): string | null {
  if (question !== "date_of_birth") return null;
  const d = isoDate(text);
  if (d === null) return "Write the date as YYYY-MM-DD.";
  if (d > today) return "A date of birth can't be in the future.";
  return null;
}

export function reduce(s: OnboardingState, e: OnboardingEvent, order: readonly OnboardingQuestion[] = ONBOARDING): OnboardingState {
  const q = order[s.step];
  const next = s.editing ? order.length : s.step + 1;
  switch (e.type) {
    case "answer": {
      if (!q) return s;
      const text = e.text.split(/\s+/).filter(Boolean).join(" ");
      const { [q.id]: _dropped, ...rest } = s.answers;
      if (!text) return { step: next, answers: rest, error: null, editing: false };
      const message = check(q.id, text, e.today);
      if (message) return { ...s, error: { question: q.id, message } };
      return { step: next, answers: { ...rest, [q.id]: text }, error: null, editing: false };
    }
    case "skip": {
      if (!q) return s;
      const { [q.id]: _dropped, ...rest } = s.answers;
      return { step: next, answers: rest, error: null, editing: false };
    }
    case "back":
      return s.editing
        ? { ...s, step: order.length, error: null, editing: false }
        : { ...s, step: Math.max(0, s.step - 1), error: null };
    case "edit":
      return { ...s, step: Math.min(Math.max(0, e.step), order.length - 1), error: null, editing: true };
  }
}

/** The records onboarding writes, all stated by the person. Skipped questions write nothing. */
export function finish(
  s: OnboardingState,
  person: PersonState,
  at: IsoDate,
  order: readonly OnboardingQuestion[] = ONBOARDING,
): PersonState {
  return order.reduce((p, q) => {
    const value = s.answers[q.id];
    if (value === undefined) return p;
    const d = q.destination;
    switch (d.store) {
      case "profile":
        return stateFact(p, d.key, value, at);
      case "relationships":
        return stateRelationship(p, value, d.kind, at);
      case "puzzles":
        return statePuzzle(p, value, at);
    }
  }, person);
}
