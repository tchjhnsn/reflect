"use client";

import { useState, useTransition } from "react";
import { PACKS } from "@/core/packs";
import type { Belief, Dated, Me } from "@/server/me-store";
import { longDate, useToday } from "@/lib/today";
import { LABELS } from "../welcome/copy";
import { Nav } from "../nav";
import { changeFact, letGoBelief, setPack, stateBelief } from "./actions";

type Result = { error: string } | { ok: true };

function Field({ id, label, history, today, hint }: { id: string; label: string; history: Dated[]; today: string | null; hint?: string }) {
  const current = history.find((h) => h.validTo === null) ?? null;
  const earlier = history.filter((h) => h !== current);
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(current?.value ?? "");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  const save = (value: string) =>
    start(async () => {
      if (!today) return;
      const r: Result = await changeFact(id, value, today);
      if ("error" in r) return setError(r.error);
      setError(null);
      setEditing(false);
    });

  return (
    <div className="me-field" data-testid={`field-${id}`}>
      <dt>{label}</dt>
      <dd>
        {editing ? (
          <form
            className="inline-edit"
            onSubmit={(e) => {
              e.preventDefault();
              save(text);
            }}
          >
            <label htmlFor={`edit-${id}`} className="sr-only">
              {label}
            </label>
            <input
              id={`edit-${id}`}
              value={text}
              placeholder={hint}
              onChange={(e) => setText(e.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? `edit-${id}-error` : undefined}
              autoFocus
            />
            {error ? (
              <p className="error" id={`edit-${id}-error`}>
                {error}
              </p>
            ) : null}
            <div className="actions">
              {current && id !== "name" ? (
                <button type="button" className="quiet" disabled={pending} onClick={() => save("")}>
                  Remove
                </button>
              ) : null}
              <button type="button" className="quiet" onClick={() => setEditing(false)}>
                Cancel
              </button>
              <button type="submit" className="primary" disabled={pending || !today}>
                Save
              </button>
            </div>
          </form>
        ) : (
          <div className="field-row">
            <span className="me-value">{current ? current.value : <span className="muted">Not stated</span>}</span>
            {current ? <span className="since">since {longDate(current.validFrom)}</span> : null}
            <button type="button" className="quiet" onClick={() => (setText(current?.value ?? ""), setEditing(true))} aria-label={`Change ${label}`}>
              {current ? "Change" : "Add"}
            </button>
          </div>
        )}
        {earlier.length > 0 ? (
          <details className="earlier">
            <summary>Earlier</summary>
            <ul>
              {earlier.map((h, i) => (
                <li key={i}>
                  {h.value} <span className="since">({longDate(h.validFrom)} to {longDate(h.validTo!)})</span>
                </li>
              ))}
            </ul>
          </details>
        ) : null}
      </dd>
    </div>
  );
}

const KINDS = {
  belief: { title: "Beliefs", about: "What you hold to be true.", placeholder: "I believe…", add: "Add belief" },
  value: { title: "Values", about: "What you hold matters.", placeholder: "What matters to me is…", add: "Add value" },
} as const;

function Beliefs({ kind, all, today }: { kind: keyof typeof KINDS; all: Belief[]; today: string | null }) {
  const k = KINDS[kind];
  const held = all.filter((b) => b.kind === kind && b.validTo === null);
  const before = all.filter((b) => b.kind === kind && b.validTo !== null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  return (
    <section className="panel" aria-labelledby={`h-${kind}`}>
      <h3 id={`h-${kind}`}>{k.title}</h3>
      <p className="note">{k.about}</p>
      {held.length === 0 ? <p className="muted">None stated yet.</p> : null}
      <ul className="beliefs">
        {held.map((b) => (
          <li key={b.id} data-testid={kind}>
            <span>{b.text}</span>
            <span className="since">since {longDate(b.validFrom)}</span>
            <button
              type="button"
              className="quiet"
              disabled={pending || !today}
              onClick={() => start(async () => void (today && (await letGoBelief(b.id, today))))}
              aria-label={`I no longer hold: ${b.text}`}
            >
              No longer
            </button>
          </li>
        ))}
      </ul>
      <form
        className="inline-edit"
        onSubmit={(e) => {
          e.preventDefault();
          start(async () => {
            if (!today) return;
            const r: Result = await stateBelief(kind, text, today);
            if ("error" in r) return setError(r.error);
            setText("");
            setError(null);
          });
        }}
      >
        <label htmlFor={`new-${kind}`} className="sr-only">
          {k.add}
        </label>
        <input id={`new-${kind}`} value={text} placeholder={k.placeholder} onChange={(e) => setText(e.target.value)} />
        {error ? <p className="error">{error}</p> : null}
        <div className="actions">
          <button type="submit" className="primary" disabled={pending || !today || text.trim() === ""}>
            {k.add}
          </button>
        </div>
      </form>
      {before.length > 0 ? (
        <details className="earlier">
          <summary>Held before</summary>
          <ul>
            {before.map((b) => (
              <li key={b.id}>
                {b.text} <span className="since">({longDate(b.validFrom)} to {longDate(b.validTo!)})</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function Packs({ chosen: saved }: { chosen: string[] }) {
  const [, start] = useTransition();
  const [chosen, setChosen] = useState(saved);
  return (
    <fieldset className="packs">
      <legend className="sr-only">Prompt packs</legend>
      {PACKS.map((p) => (
        <label key={p.id} className="pack">
          <input
            type="checkbox"
            checked={chosen.includes(p.id)}
            onChange={(e) => {
              const on = e.target.checked;
              setChosen((c) => (on ? [...c, p.id] : c.filter((x) => x !== p.id)));
              start(async () => {
                const r: Result = await setPack(p.id, on);
                if ("error" in r) setChosen((c) => (on ? c.filter((x) => x !== p.id) : [...c, p.id]));
              });
            }}
          />
          <span>
            <strong>{p.title}</strong>
            <span className="note">{p.about}</span>
          </span>
        </label>
      ))}
    </fieldset>
  );
}

export function MePage({ me }: { me: Me }) {
  const today = useToday();
  return (
    <main className="shell wide">
      <Nav current="me" />
      <h1>Me</h1>
      <p className="note">Only what you state is kept here. Nothing is guessed from your journal.</p>

      <section className="panel" aria-labelledby="h-about">
        <h2 id="h-about">About you</h2>
        <dl className="fields">
          <Field id="name" label={LABELS.name!} history={me.facts.name} today={today} />
          <Field id="date_of_birth" label={LABELS.date_of_birth!} history={me.facts.date_of_birth} today={today} hint="YYYY-MM-DD" />
          <Field id="occupation" label={LABELS.occupation!} history={me.facts.occupation} today={today} />
          <Field id="closest_friend" label={LABELS.closest_friend!} history={me.closestFriend} today={today} />
        </dl>
      </section>

      <h2 className="section-title">What you believe and value</h2>
      <p className="note">Written by you, kept apart from the facts above. Letting one go keeps it, dated, under Held before.</p>
      <Beliefs kind="belief" all={me.beliefs} today={today} />
      <Beliefs kind="value" all={me.beliefs} today={today} />

      <section className="panel" aria-labelledby="h-packs">
        <h2 id="h-packs">Prompts</h2>
        <p className="note">Choose the kinds of questions you want. The journal offers prompts only from these, and only when you ask.</p>
        <Packs chosen={me.packs} />
      </section>
    </main>
  );
}
