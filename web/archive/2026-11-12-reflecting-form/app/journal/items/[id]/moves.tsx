"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { BELIEVED, EXAMINE_STEPS, STANDS, type Believed, type Item, type ExamineStep, type ItemMove, type Stands } from "@reflect/core/items";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { moveItem } from "../actions";

const BELIEVED_HELP: Record<Believed, string> = { kept: "Kept", qualified: "True under conditions", revised: "Revised", withdrawn: "Withdrawn", suspended: "Suspended, not sure yet" };
const STANDS_HELP: Record<Stands, string> = { grounded: "Grounded in something deeper", reinforced: "More reasons for it", restated: "Said better", unchanged: "Unchanged" };

function Chips<T extends string>({ options, value, onChange, label }: { options: readonly T[]; value: T | null; onChange: (v: T) => void; label: (v: T) => string }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((o) => (
        <Button key={o} type="button" size="sm" variant={value === o ? "default" : "outline"} onClick={() => onChange(o)} aria-pressed={value === o}>{label(o)}</Button>
      ))}
    </div>
  );
}

function Section({ title, ask, children, className }: { title: string; ask?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={cn("space-y-3", className)}>
      <div>
        <h3 className="text-sm font-medium">{title}</h3>
        {ask ? <p className="text-muted-foreground text-sm">{ask}</p> : null}
      </div>
      {children}
    </section>
  );
}

/** The next moves the item's state allows, and nothing else (the statechart is in packages/core/src/items.ts). */
export function Moves({ id, item }: { id: string; item: Item }) {
  const [pending, start] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState<Record<string, string>>({});
  const [believed, setBelieved] = useState<Believed | null>(null);
  const [stands, setStands] = useState<Stands | null>(null);
  const [day, setDay] = useState("");
  const [step, setStep] = useState<ExamineStep | null>(null);
  const router = useRouter();
  const t = (k: string) => text[k] ?? "";
  const set = (k: string) => (e: React.ChangeEvent<HTMLTextAreaElement>) => setText({ ...text, [k]: e.target.value });
  const send = (move: ItemMove, clear?: string) =>
    start(async () => {
      const r = await moveItem(id, move);
      if ("error" in r) return setError(r.error);
      setError(null);
      if (clear) setText({ ...text, [clear]: "" });
      router.refresh();
    });

  const plan = (
    <Section title="Plan" ask="Decide how to carry it out.">
      <Textarea rows={2} value={t("plan")} onChange={set("plan")} placeholder="I will..." aria-label="Plan" />
      <Button size="sm" disabled={pending || !t("plan").trim()} onClick={() => send({ kind: "planned", text: t("plan") }, "plan")}>Plan it</Button>
    </Section>
  );

  let body: React.ReactNode = null;
  switch (item.state) {
    case "observed":
      body = (
        <Section title="Reflect" ask="Now that it is stated, do you still hold it?">
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={pending} onClick={() => send({ kind: "reflected", outcome: "held" })}>I hold it</Button>
            <Button size="sm" variant="outline" disabled={pending} onClick={() => send({ kind: "reflected", outcome: "unsure" })}>Unsure</Button>
            <Button size="sm" variant="outline" disabled={pending} onClick={() => send({ kind: "reflected", outcome: "dropped" })}>No longer</Button>
          </div>
          <Textarea rows={2} value={t("revised")} onChange={set("revised")} placeholder="Or say how you hold it now..." aria-label="Revised" />
          <Button size="sm" variant="outline" disabled={pending || !t("revised").trim()} onClick={() => send({ kind: "reflected", outcome: "revised", text: t("revised") }, "revised")}>Revise it</Button>
        </Section>
      );
      break;
    case "reflected":
      body = (
        <div className="space-y-6">
          <Section title="Examine" ask="Put it under scrutiny: say it precisely, ground it, set it against the strongest case, take counsel.">
            <div className="flex gap-2">
              <Button size="sm" disabled={pending} onClick={() => send({ kind: "examine" })}>Examine it</Button>
              <Button size="sm" variant="ghost" disabled={pending} onClick={() => send({ kind: "reflected", outcome: "dropped" })}>Drop it</Button>
            </div>
          </Section>
          {plan}
        </div>
      );
      break;
    case "examining":
      body = (
        <div className="space-y-6">
          <Section title="Examine" ask="Take any step, in any order, as often as it helps.">
            <ul className="divide-y rounded-lg border">
              {EXAMINE_STEPS.map((s) => {
                const notes = item.notes.filter((n) => n.step === s.id);
                return (
                  <li key={s.id} className="space-y-2 px-4 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium">{s.label}</p>
                        <p className="text-muted-foreground text-sm">{s.ask}</p>
                      </div>
                      {step === s.id ? null : <Button size="sm" variant="ghost" onClick={() => setStep(s.id)}>Write</Button>}
                    </div>
                    {notes.map((n, i) => <p key={i} className="bg-muted rounded-md px-3 py-2 text-sm whitespace-pre-wrap">{n.text}</p>)}
                    {step === s.id ? (
                      <div className="space-y-2">
                        <Textarea rows={2} autoFocus value={t(s.id)} onChange={set(s.id)} aria-label={s.label} />
                        <div className="flex gap-2">
                          <Button size="sm" disabled={pending || !t(s.id).trim()} onClick={() => { send({ kind: "examine_step", step: s.id, text: t(s.id) }, s.id); setStep(null); }}>Add</Button>
                          <Button size="sm" variant="ghost" onClick={() => setStep(null)}>Cancel</Button>
                        </div>
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </Section>
          <Section title="Settle it" ask="What you believe now, and how it stands.">
            <Chips options={BELIEVED} value={believed} onChange={setBelieved} label={(v) => BELIEVED_HELP[v]} />
            <Chips options={STANDS} value={stands} onChange={setStands} label={(v) => STANDS_HELP[v]} />
            {believed === "revised" || believed === "qualified" || stands === "restated" ? <Textarea rows={2} value={t("wording")} onChange={set("wording")} placeholder="The new wording" aria-label="New wording" /> : null}
            <div className="flex gap-2">
              <Button size="sm" disabled={pending || !believed || !stands} onClick={() => send({ kind: "examined", believed: believed!, stands: stands!, text: t("wording") || undefined }, "wording")}>Settle</Button>
              <Button size="sm" variant="ghost" disabled={pending} onClick={() => send({ kind: "another_look" })}>Needs another look</Button>
            </div>
            {item.examined ? <p className="text-muted-foreground text-sm">Settled: {BELIEVED_HELP[item.examined.believed].toLowerCase()}, {STANDS_HELP[item.examined.stands].toLowerCase()}. Examination goes on; plan when you are ready.</p> : null}
          </Section>
          {plan}
        </div>
      );
      break;
    case "planned":
    case "scheduled":
      body = (
        <div className="space-y-6">
          <Section title={item.state === "planned" ? "Act" : "Ready"} ask={item.state === "planned" ? "Put it on a day." : `Scheduled for ${item.scheduledOn}.`}>
            <div className="flex flex-wrap items-center gap-2">
              {item.state === "scheduled" ? <Button size="sm" disabled={pending} onClick={() => send({ kind: "started" })}>Start</Button> : null}
              <Input type="date" value={day} onChange={(e) => setDay(e.target.value)} className="w-44" aria-label="Day" />
              <Button size="sm" variant={item.state === "planned" ? "default" : "outline"} disabled={pending || !day} onClick={() => send({ kind: "scheduled", on: day })}>{item.state === "planned" ? "Schedule" : "Move it"}</Button>
            </div>
          </Section>
          {item.state === "planned" ? <Button size="sm" variant="ghost" disabled={pending} onClick={() => send({ kind: "examine" })}>Examine it</Button> : null}
        </div>
      );
      break;
    case "in_progress":
      body = (
        <Section title="Act" ask="How did it go?">
          <Textarea rows={2} value={t("outcome")} onChange={set("outcome")} placeholder="A note, if you like" aria-label="Note" />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={pending} onClick={() => send({ kind: "done", text: t("outcome") || undefined }, "outcome")}>Done</Button>
            <Button size="sm" variant="outline" disabled={pending} onClick={() => send({ kind: "missed", text: t("outcome") || undefined }, "outcome")}>Missed</Button>
            <Button size="sm" variant="ghost" disabled={pending} onClick={() => send({ kind: "examine" })}>Examine while acting</Button>
          </div>
        </Section>
      );
      break;
    case "missed":
      body = (
        <Section title="Observe why" ask="It was missed. What happened, and is it the same reason as before? Then go round again.">
          <Textarea rows={3} value={t("why")} onChange={set("why")} aria-label="Why it was missed" />
          <Button size="sm" disabled={pending || !t("why").trim()} onClick={() => send({ kind: "observed_why", text: t("why") }, "why")}>Go round again</Button>
        </Section>
      );
      break;
    case "done":
      body = (
        <Section title="Done" ask="Examination is ongoing: you can look at it again whenever it needs it.">
          <Button size="sm" variant="outline" disabled={pending} onClick={() => send({ kind: "examine" })}>Examine again</Button>
        </Section>
      );
      break;
    case "dropped":
      body = <p className="text-muted-foreground text-sm">Dropped. It stays in your record and leaves the pipeline.</p>;
      break;
  }
  return (
    <div className="space-y-3">
      {body}
      {error ? <p className="text-destructive text-sm" role="alert">{error}</p> : null}
    </div>
  );
}
