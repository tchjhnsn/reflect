"use client";

import { useState, useTransition } from "react";
import { PACKS } from "@/core/packs";
import type { Belief, Dated, Me } from "@/server/me-store";
import { longDate, shortDate, useToday } from "@/lib/today";
import { Nav, initialOf } from "../nav";
import { SignOut } from "../sign-out";
import { THEMES, type ThemeId } from "@/core/theme";
import { CHOICES, parseHold } from "@/core/onboarding";
import { HOLDS } from "@/core/structure";
import { CHOICE_LABELS, HOLD_LABELS, showAnswer } from "./copy";
import { changeFact, letGoBelief, setPack, setResearch, setTheme, stateBelief } from "./actions";

type Result = { error: string } | { ok: true };
type FieldId = "name" | "date_of_birth" | "occupation" | "closest_friend" | "marital_status" | "seeing_someone" | "children" | "hold_partnership" | "hold_children";

const FIELDS: { id: FieldId; label: string; blank: string; hint?: string }[] = [
  { id: "date_of_birth", label: "Born", blank: "when?", hint: "YYYY-MM-DD" },
  { id: "occupation", label: "Work", blank: "in your words" },
  { id: "closest_friend", label: "Closest friend", blank: "someone you can name" },
  { id: "marital_status", label: "Marital status", blank: "not stated" },
  { id: "seeing_someone", label: "Seeing someone", blank: "not stated" },
  { id: "children", label: "Children", blank: "not stated", hint: "0" },
];

const HOLD_FIELDS: { id: FieldId; label: string; blank: string }[] = [
  { id: "hold_partnership", label: "Partnership", blank: "how you hold it" },
  { id: "hold_children", label: "Having children", blank: "how you hold it" },
];

function show(id: FieldId, value: string): string {
  return id === "date_of_birth" ? longDate(value) : showAnswer(id, value);
}

const ALONE = ["not_for_me", "unsure"];

function Editor({ id, label, history, today, hint, onClose }: { id: FieldId; label: string; history: Dated[]; today: string | null; hint?: string; onClose: () => void }) {
  const current = history.find((h) => h.validTo === null) ?? null;
  const earlier = history.filter((h) => h !== current);
  const isHold = id.startsWith("hold_");
  const choices = CHOICES[id];
  const startHold = isHold && current ? parseHold(current.value) : { ids: [] as string[], words: "" };
  const [text, setText] = useState(isHold ? startHold.words : (current?.value ?? ""));
  const [picks, setPicks] = useState<string[]>(startHold.ids);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const toggle = (h: string) =>
    setPicks((p) => (p.includes(h) ? p.filter((x) => x !== h) : ALONE.includes(h) ? [h] : [...p.filter((x) => !ALONE.includes(x)), h]));
  const composed = () => (isHold ? (picks.length ? picks.join(",") + (text.trim() ? `; ${text}` : "") : "") : text);
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
        save(composed());
      }}
    >
      {choices ? (
        <>
          <p id={`edit-${id}-label`} className="lbl">
            {label}
          </p>
          <div className="choices" role="radiogroup" aria-labelledby={`edit-${id}-label`}>
            {choices.map((c) => (
              <button key={c} type="button" role="radio" aria-checked={text === c} className="chip" onClick={() => setText(c)}>
                {CHOICE_LABELS[c] ?? c}
              </button>
            ))}
          </div>
        </>
      ) : isHold ? (
        <>
          <p id={`edit-${id}-label`} className="lbl">
            {label}
          </p>
          <div className="choices" role="group" aria-labelledby={`edit-${id}-label`}>
            {HOLDS.map((h) => (
              <button key={h.id} type="button" aria-pressed={picks.includes(h.id)} className="chip" onClick={() => toggle(h.id)}>
                {HOLD_LABELS[h.id] ?? h.id}
              </button>
            ))}
          </div>
          <label htmlFor={`edit-${id}`} className="sr-only">
            In your own words
          </label>
          <input id={`edit-${id}`} value={text} placeholder="In your own words, if you like" onChange={(e) => setText(e.target.value)} />
        </>
      ) : (
        <>
          <label htmlFor={`edit-${id}`} className="lbl">
            {label}
          </label>
          <input
            id={`edit-${id}`}
            value={text}
            placeholder={hint}
            inputMode={id === "children" ? "numeric" : undefined}
            onChange={(e) => setText(e.target.value)}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? `edit-${id}-error` : undefined}
            autoFocus
          />
        </>
      )}
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

function Research({ allowed }: { allowed: boolean }) {
  const [on, setOn] = useState(allowed);
  const [, start] = useTransition();
  const flip = () => {
    const next = !on;
    setOn(next);
    start(async () => {
      const r: Result = await setResearch(next);
      if ("error" in r) setOn(!next);
    });
  };
  return (
    <div className="choices" role="group" aria-label="Improving the reader">
      <button type="button" className="chip" aria-pressed={on} onClick={flip}>
        Let my labels improve the reader
      </button>
    </div>
  );
}

export function MePage({ me, theme, research }: { me: Me; theme: ThemeId; research: boolean }) {
  const today = useToday();
  const [editing, setEditing] = useState<FieldId | null>(null);
  const name = me.facts.name.find((h) => h.validTo === null)?.value ?? null;
  const history = (id: FieldId): Dated[] =>
    id === "closest_friend" ? me.closestFriend : id === "hold_partnership" ? me.holds.partnership : id === "hold_children" ? me.holds.children : me.facts[id];
  const rows = (fields: typeof FIELDS) =>
    fields.map((f) => {
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
    });

  return (
    <main className="page">
      <Nav current="me" eyebrow="About you" initial={initialOf(name)} />
      <h1 className="title">
        <button type="button" onClick={() => setEditing("name")} aria-label="Change Name" style={{ font: "inherit", textAlign: "left" }}>
          {name ?? "You"}
        </button>
      </h1>
      {editing === "name" ? <Editor id="name" label="Name" history={me.facts.name} today={today} onClose={() => setEditing(null)} /> : null}

      <dl className="facts">{rows(FIELDS)}</dl>
      <p className="sec-note">Only what you state is kept here. Nothing is guessed from your journal.</p>

      <section className="sec" aria-labelledby="h-holds">
        <div className="sec-head">
          <h2 id="h-holds" className="lbl">
            How I hold family
          </h2>
        </div>
        <p className="sec-note">Wanting, working toward, feeling you ought to, and holding sacred are kept apart.</p>
        <dl className="facts" style={{ borderTop: 0 }}>
          {rows(HOLD_FIELDS)}
        </dl>
      </section>

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
        <p className="sec-note">
          When you reflect, a model guesses which elements were part of your entry and you correct it. If you allow it, your corrections (never
          your words) help that guess get better. Off unless you turn it on.
        </p>
        <Research allowed={research} />
      </section>

      <div className="foot">
        <span className="note">Reflect is private to you.</span>
        <SignOut />
      </div>
    </main>
  );
}
