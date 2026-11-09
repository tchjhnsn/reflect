"use client";

import { useState, useTransition } from "react";
import { PACKS } from "@/core/packs";
import type { Belief, Dated, Me } from "@/server/me-store";
import { longDate, shortDate, useToday } from "@/lib/today";
import { Nav, initialOf } from "../nav";
import { SignOut } from "../sign-out";
import { THEMES, type ThemeId } from "@/core/theme";
import { changeFact, letGoBelief, setPack, setTheme, stateBelief } from "./actions";

type Result = { error: string } | { ok: true };
type FieldId = "name" | "date_of_birth" | "occupation" | "closest_friend";

const FIELDS: { id: FieldId; label: string; blank: string; hint?: string }[] = [
  { id: "date_of_birth", label: "Born", blank: "when?", hint: "YYYY-MM-DD" },
  { id: "occupation", label: "Work", blank: "in your words" },
  { id: "closest_friend", label: "Closest friend", blank: "someone you can name" },
];

function show(id: FieldId, value: string): string {
  return id === "date_of_birth" ? longDate(value) : value;
}

function Editor({ id, label, history, today, hint, onClose }: { id: FieldId; label: string; history: Dated[]; today: string | null; hint?: string; onClose: () => void }) {
  const current = history.find((h) => h.validTo === null) ?? null;
  const earlier = history.filter((h) => h !== current);
  const [text, setText] = useState(current?.value ?? "");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const save = (value: string) =>
    start(async () => {
      if (!today) return;
      const r: Result = await changeFact(id, value, today);
      if ("error" in r) return setError(r.error);
      onClose();
    });

  return (
    <form
      className="editor"
      data-testid={`edit-${id}`}
      onSubmit={(e) => {
        e.preventDefault();
        save(text);
      }}
    >
      <label htmlFor={`edit-${id}`} className="lbl">
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
        <button type="button" className="quiet" onClick={onClose}>
          Cancel
        </button>
        <button type="submit" className="primary" disabled={pending || !today}>
          Save
        </button>
      </div>
      {current ? <p className="sec-note">Saving keeps the old one, dated. Since {longDate(current.validFrom)}.</p> : null}
      {earlier.length > 0 ? (
        <ul className="history" aria-label={`Earlier ${label}`}>
          {earlier.map((h, i) => (
            <li key={i}>
              {show(id, h.value)} <span>{shortDate(h.validFrom)} to {shortDate(h.validTo!)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </form>
  );
}

const KINDS = {
  belief: { title: "I believe", placeholder: "Something else I believe…", add: "Add belief" },
  value: { title: "What matters to me", placeholder: "Something else that matters…", add: "Add value" },
} as const;

function Beliefs({ kind, all, today }: { kind: keyof typeof KINDS; all: Belief[]; today: string | null }) {
  const k = KINDS[kind];
  const held = all.filter((b) => b.kind === kind && b.validTo === null);
  const before = all.filter((b) => b.kind === kind && b.validTo !== null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  return (
    <section className="sec" aria-labelledby={`h-${kind}`}>
      <div className="sec-head">
        <h2 id={`h-${kind}`} className="lbl">
          {k.title}
        </h2>
      </div>
      <ul className="held">
        {held.map((b) => (
          <li key={b.id} data-testid={kind}>
            <span>{b.text}</span>
            <span className="since">since {shortDate(b.validFrom)}</span>
            <button
              type="button"
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
        className="add"
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
        {text.trim() ? (
          <div className="actions">
            <button type="submit" className="primary" disabled={pending || !today}>
              {k.add}
            </button>
          </div>
        ) : null}
      </form>
      {before.length > 0 ? (
        <details className="earlier">
          <summary>Held before</summary>
          <ul className="history">
            {before.map((b) => (
              <li key={b.id}>
                {b.text} <span>{shortDate(b.validFrom)} to {shortDate(b.validTo!)}</span>
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
  const toggle = (id: string) => {
    const on = !chosen.includes(id);
    setChosen((c) => (on ? [...c, id] : c.filter((x) => x !== id)));
    start(async () => {
      const r: Result = await setPack(id, on);
      if ("error" in r) setChosen((c) => (on ? c.filter((x) => x !== id) : [...c, id]));
    });
  };
  return (
    <div className="chips" role="group" aria-label="Prompt packs">
      {PACKS.map((p) => (
        <button key={p.id} type="button" className="chip" aria-pressed={chosen.includes(p.id)} title={p.about} onClick={() => toggle(p.id)}>
          {p.title}
        </button>
      ))}
    </div>
  );
}

function Themes({ current }: { current: ThemeId }) {
  const [theme, setLocal] = useState<ThemeId>(current);
  const [, start] = useTransition();
  const pick = (id: ThemeId) => {
    const before = theme;
    setLocal(id);
    document.documentElement.dataset.theme = id;
    start(async () => {
      const r: Result = await setTheme(id);
      if ("error" in r) {
        setLocal(before);
        document.documentElement.dataset.theme = before;
      }
    });
  };
  return (
    <div className="themes" role="group" aria-label="Theme">
      {THEMES.map((t) => (
        <button key={t.id} type="button" className="theme" aria-pressed={theme === t.id} title={t.about} onClick={() => pick(t.id)}>
          <span className={`swatch ${t.id}`} aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          {t.title}
        </button>
      ))}
    </div>
  );
}

export function MePage({ me, theme }: { me: Me; theme: ThemeId }) {
  const today = useToday();
  const [editing, setEditing] = useState<FieldId | null>(null);
  const name = me.facts.name.find((h) => h.validTo === null)?.value ?? null;
  const history = (id: FieldId): Dated[] => (id === "closest_friend" ? me.closestFriend : me.facts[id]);

  return (
    <main className="page">
      <Nav current="me" eyebrow="About you" initial={initialOf(name)} />
      <h1 className="title">
        <button type="button" onClick={() => setEditing("name")} aria-label="Change Name" style={{ font: "inherit", textAlign: "left" }}>
          {name ?? "You"}
        </button>
      </h1>
      {editing === "name" ? <Editor id="name" label="Name" history={me.facts.name} today={today} onClose={() => setEditing(null)} /> : null}

      <dl className="facts">
        {FIELDS.map((f) => {
          const current = history(f.id).find((h) => h.validTo === null) ?? null;
          return editing === f.id ? (
            <Editor key={f.id} id={f.id} label={f.label} history={history(f.id)} today={today} hint={f.hint} onClose={() => setEditing(null)} />
          ) : (
            <div key={f.id} className="fact" data-testid={`field-${f.id}`}>
              <dt>{f.label}</dt>
              <dd>
                <button type="button" className={`fact-value${current ? "" : " blank"}`} onClick={() => setEditing(f.id)} aria-label={`Change ${f.label}`}>
                  {current ? show(f.id, current.value) : f.blank}
                </button>
              </dd>
            </div>
          );
        })}
      </dl>
      <p className="sec-note">Only what you state is kept here. Nothing is guessed from your journal.</p>

      <Beliefs kind="belief" all={me.beliefs} today={today} />
      <Beliefs kind="value" all={me.beliefs} today={today} />

      <section className="sec" aria-labelledby="h-packs">
        <div className="sec-head">
          <h2 id="h-packs" className="lbl">
            Questions I want
          </h2>
        </div>
        <p className="sec-note">The journal offers a question only from these, and only when you ask.</p>
        <Packs chosen={me.packs} />
      </section>

      <section className="sec" aria-labelledby="h-settings">
        <div className="sec-head">
          <h2 id="h-settings" className="lbl">
            Settings
          </h2>
        </div>
        <p className="sec-note">Theme. Only the look changes; what you write stays the same.</p>
        <Themes current={theme} />
      </section>

      <div className="foot">
        <span className="note">Reflect is private to you.</span>
        <SignOut />
      </div>
    </main>
  );
}
