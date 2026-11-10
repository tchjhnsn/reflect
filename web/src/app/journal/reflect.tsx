"use client";

import { useEffect, useState, useTransition } from "react";
import { FRAMES, frameById, type FrameId } from "@/core/reflect";
import type { Reflection } from "@/core/journal";
import { CATEGORIES, GUESS_AT, PRIMES, elementName, type Salience } from "@/core/state";
import { beginReflection, guessLabels, reflectAnswer, saveLabels } from "./actions";

type Term = Omit<Salience["selfTerms"][number], "basis"> & { basis: string | null; modelBasis: string };

/**
 * The first step of reflecting: the person labels what was part of the entry, using the elements as
 * tools. The model's guesses arrive as a question and pre-mark chips; the person confirms, removes or
 * adds. Nothing the model guessed is kept unless the person leaves it marked.
 */
export function Labels({ entryId, onDone }: { entryId: string; onDone: () => void }) {
  const [loading, setLoading] = useState(true);
  const [weights, setWeights] = useState<Record<string, number> | null>(null);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [terms, setTerms] = useState<Term[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  useEffect(() => {
    let live = true;
    guessLabels(entryId).then(({ salience }) => {
      if (!live) return;
      if (salience) {
        setWeights(salience.weights);
        setChosen(new Set(PRIMES.filter((p) => salience.weights[p]! >= GUESS_AT)));
        setTerms(salience.selfTerms.map((t) => ({ ...t, modelBasis: t.basis, basis: t.basis })));
      }
      setLoading(false);
    });
    return () => {
      live = false;
    };
  }, [entryId]);

  const guessed = weights ? PRIMES.filter((p) => weights[p]! >= GUESS_AT).sort((a, b) => weights[b]! - weights[a]!) : [];
  const toggle = (p: string) => setChosen((c) => { const n = new Set(c); if (n.has(p)) n.delete(p); else n.add(p); return n; });
  const keep = () =>
    start(async () => {
      const r = await saveLabels(entryId, weights, [...chosen], terms);
      if ("error" in r) return setError(r.error);
      onDone();
    });

  return (
    <div className="reflect labels" role="group" aria-labelledby={`labels-${entryId}`}>
      <p id={`labels-${entryId}`} className="reflect-title">
        What was part of this?
      </p>
      <p className="label-guess" aria-live="polite">
        {loading
          ? "Reading your entry…"
          : guessed.length > 0
            ? `I noticed ${list(guessed.map(elementName))}. Is that right? Change anything that isn’t.`
            : "Mark anything that was part of it. Leave the rest."}
      </p>
      <div className="element-grid">
        {CATEGORIES.map((c) => (
          <div key={c.id} className="element-pair" role="group" aria-label={c.id}>
            {c.pair.map((p) => (
              <button
                key={p}
                type="button"
                className={`chip${weights && weights[p]! >= GUESS_AT ? " guessed" : ""}`}
                aria-pressed={chosen.has(p)}
                onClick={() => toggle(p)}
                disabled={loading}
              >
                {elementName(p)}
              </button>
            ))}
          </div>
        ))}
      </div>
      {terms.map((t, i) => (
        <div key={t.quote} className="self-term">
          <label htmlFor={`term-${entryId}-${i}`}>
            {t.whose === "self" ? "You called yourself" : "You called someone"} &ldquo;{t.quote}&rdquo;. What is that comparison made through?
          </label>
          <select
            id={`term-${entryId}-${i}`}
            value={t.basis ?? ""}
            onChange={(e) => setTerms((ts) => ts.map((x, j) => (j === i ? { ...x, basis: e.target.value || null } : x)))}
          >
            {PRIMES.filter((p) => p !== "self-differentiation").map((p) => (
              <option key={p} value={p}>
                {elementName(p)}
                {p === t.modelBasis ? " (my guess)" : ""}
              </option>
            ))}
            <option value="">It isn&rsquo;t a comparison</option>
          </select>
        </div>
      ))}
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <button type="button" className="quiet" onClick={onDone} disabled={pending}>
          Skip
        </button>
        <button type="button" className="primary" onClick={keep} disabled={pending || loading}>
          Keep labels
        </button>
      </div>
    </div>
  );
}

function list(xs: string[]): string {
  return xs.length <= 1 ? (xs[0] ?? "") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`;
}

/** Pick a frame, then fixed open questions one at a time. Skipping writes nothing. No model is asked. */
export function Reflect({ entryId, onClose, onStart }: { entryId: string; onClose: () => void; onStart: (id: string) => void }) {
  const [labeled, setLabeled] = useState(false);
  const [frame, setFrame] = useState<FrameId | null>(null);
  const [id, setId] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const f = frame ? frameById(frame) : null;
  const q = f?.questions[step] ?? null;

  const pick = (fid: FrameId) =>
    start(async () => {
      const r = await beginReflection(entryId, fid);
      if ("error" in r) return setError(r.error);
      setError(null);
      setFrame(fid);
      setId(r.id);
      onStart(r.id);
    });

  const next = (words: string) =>
    start(async () => {
      if (!id) return;
      const r = await reflectAnswer(id, step, words);
      if ("error" in r) return setError(r.error);
      setError(null);
      setText("");
      setStep(step + 1);
    });

  if (!labeled) return <Labels entryId={entryId} onDone={() => setLabeled(true)} />;

  if (!f || !id) {
    return (
      <div className="reflect" role="group" aria-labelledby={`reflect-${entryId}`}>
        <p id={`reflect-${entryId}`} className="reflect-title">
          What would you like to look at?
        </p>
        <div className="choices">
          {FRAMES.map((fr) => (
            <button key={fr.id} type="button" className="chip" onClick={() => pick(fr.id)} disabled={pending}>
              {fr.title}
              <span className="chip-note">{fr.about}</span>
            </button>
          ))}
        </div>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="actions">
          <button type="button" className="quiet" onClick={onClose}>
            Not now
          </button>
        </div>
      </div>
    );
  }

  if (!q) {
    return (
      <div className="reflect">
        <p className="reflect-title">That&rsquo;s every question for {f.title.toLowerCase()}.</p>
        <div className="actions">
          <button type="button" className="primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <form
      className="reflect"
      onSubmit={(e) => {
        e.preventDefault();
        next(text);
      }}
    >
      <p className="reflect-step">
        {f.title} &middot; {step + 1} of {f.questions.length}
      </p>
      <label htmlFor={`reflect-q-${entryId}`} className="more-prompt">
        {q.text}
      </label>
      <textarea id={`reflect-q-${entryId}`} value={text} rows={3} onChange={(e) => setText(e.target.value)} autoFocus />
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <button type="button" className="quiet" onClick={onClose} disabled={pending}>
          Stop here
        </button>
        <button type="button" className="quiet" onClick={() => next("")} disabled={pending}>
          Skip
        </button>
        <button type="submit" className="primary" disabled={pending || text.trim() === ""}>
          Next
        </button>
      </div>
    </form>
  );
}

/** A kept reflection: what happened apart from what the person felt they ought. */
export function Reflections({ items }: { items: readonly Reflection[] }) {
  const shown = items.filter((r) => r.answers.length > 0);
  if (shown.length === 0) return null;
  return (
    <div className="reflections">
      {shown.map((r) => {
        const is = r.answers.filter((a) => a.stance === "descriptive");
        const ought = r.answers.filter((a) => a.stance === "normative");
        return (
          <section key={r.id} className="reflection" data-testid="reflection" aria-label={`Reflection: ${frameById(r.frame)?.title ?? r.frame}`}>
            <p className="reflect-step">Reflected on {frameById(r.frame)?.title.toLowerCase() ?? r.frame}</p>
            {is.length > 0 ? (
              <dl data-testid="reflection-is">
                {is.map((a) => (
                  <div key={a.seq}>
                    <dt>{a.question}</dt>
                    <dd>{a.text}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
            {ought.length > 0 ? (
              <>
                <p className="reflect-step">What you felt you ought</p>
                <dl data-testid="reflection-ought">
                  {ought.map((a) => (
                    <div key={a.seq}>
                      <dt>{a.question}</dt>
                      <dd>{a.text}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
