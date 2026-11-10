"use client";

import { useState, useTransition } from "react";
import { BIOGRAPHY_SECTIONS, PROFILE_PROMPT, parseProfileReply, type ProfileReport, type Sourced } from "@/core/profile-prompt";
import { longDate, useToday } from "@/lib/today";
import { showAnswer } from "./copy";
import { changeFact, stateBelief } from "./actions";

const FACT_LABEL: Record<string, string> = { name: "Name", date_of_birth: "Born", occupation: "Work", marital_status: "Marital status", children: "Children" };

function Source({ s }: { s: Sourced }) {
  return (
    <p className="source">
      <span className={`basis ${s.basis}`}>{s.basis === "i-said" ? "You told it" : "Its inference"}</span> · {s.confidence} confidence
      {s.evidence ? <> · {s.evidence}</> : null}
    </p>
  );
}

/** "This is mine" turns one reported item into the person's own statement. */
function Keep({ label, keep }: { label: string; keep: () => Promise<{ error: string } | { ok: true }> }) {
  const [state, setState] = useState<"idle" | "kept" | string>("idle");
  const [pending, start] = useTransition();
  if (state === "kept") return <span className="kept">Kept as yours</span>;
  return (
    <>
      <button type="button" className="quiet keep" disabled={pending} aria-label={`This is mine: ${label}`} onClick={() => start(async () => {
        const r = await keep();
        setState("error" in r ? r.error : "kept");
      })}>
        This is mine
      </button>
      {state !== "idle" ? <span className="error">{state}</span> : null}
    </>
  );
}

function Report({ report, onClear }: { report: ProfileReport; onClear: () => void }) {
  const today = useToday();
  const at = today ?? "";
  const empty = !report.facts.length && !report.biography.length && !report.beliefs.length && !report.values.length && !report.puzzle;
  return (
    <div className="ai-report" data-testid="ai-report">
      <p className="lbl">
        What {report.assistant} said{report.reportedAt ? `, ${longDate(report.reportedAt)}` : ""}
      </p>
      {report.drewOn ? <p className="sec-note">It drew on: {report.drewOn}.</p> : null}
      {empty ? <p className="sec-note">It didn&rsquo;t report anything it knows about you.</p> : null}

      {report.puzzle ? (
        <div className="ai-block">
          <p className="lbl">The question you keep turning over</p>
          <p className="puzzle">{report.puzzle.statement}</p>
          {report.puzzle.why ? <p className="sec-note">{report.puzzle.why}</p> : null}
          <p className="source">{report.puzzle.confidence} confidence</p>
          {report.puzzle.alternatives.length ? (
            <ul className="history">
              {report.puzzle.alternatives.map((a) => (
                <li key={a}>Or: {a}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {report.facts.length ? (
        <div className="ai-block">
          <p className="lbl">Facts</p>
          <ul className="ai-list">
            {report.facts.map((f) => (
              <li key={f.key}>
                <span>
                  {FACT_LABEL[f.key]}: {f.key === "date_of_birth" ? longDate(f.value) : showAnswer(f.key, f.value)}
                </span>
                <Keep label={`${FACT_LABEL[f.key]} ${f.value}`} keep={() => changeFact(f.key, f.value, at)} />
                <Source s={f} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {BIOGRAPHY_SECTIONS.map((sec) => {
        const parts = report.biography.filter((b) => b.section === sec.id);
        return parts.length ? (
          <div key={sec.id} className="ai-block">
            <p className="lbl">{sec.title}</p>
            <ul className="ai-list">
              {parts.map((b, i) => (
                <li key={i}>
                  <span>
                    {b.when ? <em>{b.when}. </em> : null}
                    {b.text}
                  </span>
                  <Source s={b} />
                </li>
              ))}
            </ul>
          </div>
        ) : null;
      })}

      {(["beliefs", "values"] as const).map((k) =>
        report[k].length ? (
          <div key={k} className="ai-block">
            <p className="lbl">{k === "beliefs" ? "What you believe" : "What matters to you"}</p>
            <ul className="ai-list">
              {report[k].map((b) => (
                <li key={b.text}>
                  <span>{b.text}</span>
                  <Keep label={b.text} keep={() => stateBelief(k === "beliefs" ? "belief" : "value", b.text, at)} />
                  <Source s={b} />
                </li>
              ))}
            </ul>
          </div>
        ) : null,
      )}

      {report.openQuestions.length ? (
        <div className="ai-block">
          <p className="lbl">What it couldn&rsquo;t tell</p>
          <ul className="history">
            {report.openQuestions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {report.note ? <p className="sec-note ai-note">{report.note}</p> : null}
      {report.dropped ? (
        <p className="sec-note">
          {report.dropped} {report.dropped === 1 ? "entry was" : "entries were"} left out because {report.dropped === 1 ? "it" : "they"} didn&rsquo;t say
          where {report.dropped === 1 ? "it" : "they"} came from or didn&rsquo;t fit.
        </p>
      ) : null}
      <p className="sec-note">
        The life story is shown, not kept: there is no place for it in your profile yet. Facts, beliefs and values are kept only when you mark
        them as yours.
      </p>
      <div className="actions">
        <button type="button" className="quiet" onClick={onClear}>
          Clear
        </button>
      </div>
    </div>
  );
}

/** Ask the AI you use most what it knows about you, then see it here. Its answer is never kept as is. */
export function AskYourAi() {
  const [reply, setReply] = useState("");
  const [report, setReport] = useState<ProfileReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(PROFILE_PROMPT);
      setCopied(true);
    } catch {
      setError("Couldn't copy. Open the prompt below and copy it by hand.");
    }
  };
  const show = () => {
    const r = parseProfileReply(reply);
    if ("error" in r) return setError(r.error);
    setError(null);
    setReport(r.report);
    setReply("");
  };

  return (
    <section className="sec" aria-labelledby="h-ask-ai">
      <div className="sec-head">
        <h2 id="h-ask-ai" className="lbl">
          What your AI knows about you
        </h2>
      </div>
      <p className="sec-note">
        Give this prompt to the AI you talk to most. It answers only from what it already knows about you, says where each thing comes from, and
        leaves out what it doesn&rsquo;t know. Paste its answer here to read it.
      </p>
      <div className="actions" style={{ justifyContent: "flex-start" }}>
        <button type="button" className="primary" onClick={copy}>
          {copied ? "Copied" : "Copy the prompt"}
        </button>
      </div>
      <details className="earlier">
        <summary>Read the prompt</summary>
        <pre className="prompt-text" tabIndex={0} aria-label="The prompt">{PROFILE_PROMPT}</pre>
      </details>
      {report ? (
        <Report report={report} onClear={() => setReport(null)} />
      ) : (
        <form
          className="add"
          onSubmit={(e) => {
            e.preventDefault();
            show();
          }}
        >
          <label htmlFor="ai-reply" className="sr-only">
            Paste its answer
          </label>
          <textarea id="ai-reply" rows={4} value={reply} placeholder="Paste its answer…" onChange={(e) => setReply(e.target.value)} />
          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
          {reply.trim() ? (
            <div className="actions">
              <button type="submit" className="primary">
                Show what it said
              </button>
            </div>
          ) : null}
        </form>
      )}
    </section>
  );
}
