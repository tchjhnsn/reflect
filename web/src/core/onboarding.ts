// Mirrors lean/NTS/Onboarding.lean and thrivesight_core/nts/onboarding.py. No model runs.
import { ONBOARDING, type OnboardingQuestion } from "./structure";
import { isoDate, stateFact, statePuzzle, stateRelationship, type IsoDate, type PersonState } from "./person";

export type OnboardingState = {
  step: number;
  answers: Readonly<Record<string, string>>;
  error: { question: string; message: string } | null;
};

export type OnboardingEvent =
  | { type: "answer"; text: string; today: IsoDate }
  | { type: "skip" }
  | { type: "back" }
  | { type: "goto"; step: number };

export const START: OnboardingState = { step: 0, answers: {}, error: null };

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
  switch (e.type) {
    case "answer": {
      if (!q) return s;
      const text = e.text.split(/\s+/).filter(Boolean).join(" ");
      const { [q.id]: _dropped, ...rest } = s.answers;
      if (!text) return { step: s.step + 1, answers: rest, error: null };
      const message = check(q.id, text, e.today);
      if (message) return { ...s, error: { question: q.id, message } };
      return { step: s.step + 1, answers: { ...rest, [q.id]: text }, error: null };
    }
    case "skip": {
      if (!q) return s;
      const { [q.id]: _dropped, ...rest } = s.answers;
      return { step: s.step + 1, answers: rest, error: null };
    }
    case "back":
      return { ...s, step: Math.max(0, s.step - 1), error: null };
    case "goto":
      return { ...s, step: Math.min(Math.max(0, e.step), order.length), error: null };
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
