"use client";

import { ONBOARDING } from "@/core/structure";
import { LABELS } from "./copy";

type Props = {
  answers: Readonly<Record<string, string>>;
  onEdit: (step: number) => void;
  onRemove: (step: number) => void;
  onBack: () => void;
  onConfirm: () => void;
  pending?: boolean;
  error?: string | null;
};

export function Review({ answers, onEdit, onRemove, onBack, onConfirm, pending = false, error = null }: Props) {
  const given = ONBOARDING.map((q, step) => ({ q, step, value: answers[q.id] })).filter((x) => x.value !== undefined);

  return (
    <section className="screen">
      <h1 className="prompt">Here&rsquo;s what you told us.</h1>
      {given.length === 0 ? (
        <p className="note" data-testid="nothing">
          You skipped everything, so nothing will be kept.
        </p>
      ) : (
        <ul className="cards">
          {given.map(({ q, step, value }) => (
            <li key={q.id} className="card" data-testid={`card-${q.id}`}>
              <span className="label">{LABELS[q.id] ?? q.id}</span>
              <span className="value">{value}</span>
              <span className="card-actions">
                <button type="button" className="quiet" onClick={() => onEdit(step)}>
                  Edit
                </button>
                <button type="button" className="quiet" onClick={() => onRemove(step)} aria-label={`Remove ${LABELS[q.id]}`}>
                  Remove
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="note">Nothing is kept until you say so.</p>
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <button type="button" className="quiet" onClick={onBack}>
          Back
        </button>
        <button type="button" className="primary" onClick={onConfirm} disabled={pending}>
          {given.length === 0 ? "Continue" : "Keep these"}
        </button>
      </div>
    </section>
  );
}
