"use client";

import { useReducer, useState } from "react";
import { ONBOARDING } from "@/core/structure";
import { EMPTY_PERSON, type IsoDate, type PersonState } from "@/core/person";
import { START, currentQuestion, finish, isDone, reduce, type OnboardingEvent, type OnboardingState } from "@/core/onboarding";
import { Question } from "./question";
import { Review } from "./review";

function today(): IsoDate {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` as IsoDate;
}

export function Onboarding() {
  const [state, dispatch] = useReducer((s: OnboardingState, e: OnboardingEvent) => reduce(s, e), START);
  const [saved, setSaved] = useState<PersonState | null>(null);
  const q = currentQuestion(state);

  if (saved) {
    const count = saved.facts.length + saved.relationships.length;
    return (
      <main className="shell">
        <section className="screen" aria-live="polite">
          <p className="eyebrow">Reflect</p>
          <h1 className="prompt">You&rsquo;re set.</h1>
          <p className="note" data-testid="saved-count">
            {count === 0 ? "Nothing was kept. You can add any of it later." : `Kept ${count} of your answers.`}
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <ol className="dots" aria-label={`Question ${Math.min(state.step + 1, ONBOARDING.length)} of ${ONBOARDING.length}`}>
        {ONBOARDING.map((item, i) => (
          <li key={item.id} className={i < state.step ? "done" : i === state.step ? "now" : ""} />
        ))}
      </ol>
      {isDone(state) || !q ? (
        <Review
          answers={state.answers}
          onEdit={(step) => dispatch({ type: "edit", step })}
          onRemove={(step) => {
            dispatch({ type: "edit", step });
            dispatch({ type: "skip" });
          }}
          onBack={() => dispatch({ type: "back" })}
          onConfirm={() => setSaved(finish(state, EMPTY_PERSON, today()))}
        />
      ) : (
        <Question
          key={q.id}
          id={q.id}
          initial={state.answers[q.id] ?? ""}
          error={state.error?.question === q.id ? state.error.message : null}
          today={today()}
          canGoBack={state.step > 0 || state.editing}
          onAnswer={(text) => dispatch({ type: "answer", text, today: today() })}
          onSkip={() => dispatch({ type: "skip" })}
          onBack={() => dispatch({ type: "back" })}
        />
      )}
    </main>
  );
}
