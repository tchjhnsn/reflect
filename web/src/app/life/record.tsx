import Link from "next/link";
import { PenLineIcon, Trash2Icon } from "lucide-react";
import { ORIGINS } from "@reflect/core/origins";
import { shown, titleOf, type Body, type Field, type Kind, type LifeAppId } from "@reflect/core/life";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { LifeAct, LifeRecord } from "@/server/life-store";
import { actLifeAction, addLifeAction, removeLifeAction, saveLifeAction, stateLifeAction } from "./actions";

const SELECT = "border-input bg-background h-9 w-full rounded-md border px-3 text-sm shadow-xs";
const today = () => new Date().toISOString().slice(0, 10);

/** One field's input, filled with what the record holds. */
function FieldInput({ f, body }: { f: Field; body: Body }) {
  const v = body[f.key];
  const id = `f-${f.key}`;
  if (f.kind === "origins") {
    const on = typeof v === "string" ? v.split(",") : [];
    return (
      <fieldset className="grid gap-1.5 sm:col-span-2">
        <legend className="text-sm font-medium">{f.label}</legend>
        <div className="flex flex-wrap gap-2">
          {ORIGINS.map((o) => (
            <label key={o.id} className="has-[:checked]:bg-foreground has-[:checked]:text-background cursor-pointer rounded-full border px-3 py-1 text-sm">
              <input type="checkbox" name={f.key} value={o.id} defaultChecked={on.includes(o.id)} className="sr-only" />{o.label}
            </label>
          ))}
        </div>
      </fieldset>
    );
  }
  if (f.kind === "check") {
    return (
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" name={f.key} defaultChecked={v === true} className="size-4 accent-current" />
        {f.label}
      </label>
    );
  }
  return (
    <div className={cn("grid gap-1.5", f.kind === "long" ? "sm:col-span-2" : "")}>
      <label htmlFor={id} className="text-sm font-medium">{f.label}</label>
      {f.kind === "long" ? <Textarea id={id} name={f.key} defaultValue={typeof v === "string" ? v : ""} placeholder={f.hint} rows={3} />
        : f.kind === "choice" ? (
          <select id={id} name={f.key} defaultValue={typeof v === "string" ? v : ""} className={SELECT}>
            <option value="">Not said</option>
            {f.options!.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        )
        : <Input id={id} name={f.key} type={f.kind === "day" ? "date" : f.kind === "number" || f.kind === "money" ? "text" : "text"} inputMode={f.kind === "number" || f.kind === "money" ? "decimal" : undefined} defaultValue={v === undefined ? "" : String(v)} placeholder={f.kind === "money" ? "$" : f.hint} />}
    </div>
  );
}

/** The fields of a kind as a form: a new record when none is given, else the record's own. */
export function RecordForm({ app, kind, record, back, submit }: { app: LifeAppId; kind: Kind; record?: LifeRecord; back: string; submit?: string }) {
  const action = record ? saveLifeAction.bind(null, record.id) : addLifeAction.bind(null, app, kind.id);
  const checks = kind.fields.filter((f) => f.kind === "check");
  return (
    <form action={action} className="grid gap-4">
      <input type="hidden" name="$back" value={back} />
      {kind.hidden && typeof record?.body.ref === "string" ? <input type="hidden" name="ref" value={record.body.ref} /> : null}
      <div className="grid gap-4 sm:grid-cols-2">
        {kind.fields.filter((f) => f.kind !== "check" && !(kind.hidden && f.key === "ref")).map((f) => <FieldInput key={f.key} f={f} body={record?.body ?? {}} />)}
      </div>
      {checks.length ? <div className="grid gap-2 sm:grid-cols-2">{checks.map((f) => <FieldInput key={f.key} f={f} body={record?.body ?? {}} />)}</div> : null}
      <div><Button type="submit">{submit ?? (record ? "Save" : `Keep ${kind.one ? "it" : `this ${kind.label.toLowerCase()}`}`)}</Button></div>
    </form>
  );
}

/** A record's states as buttons, the one it is in pressed. Each press is kept. */
export function States({ kind, record, back }: { kind: Kind; record: LifeRecord; back: string }) {
  if (!kind.states?.length) return null;
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="State">
      {kind.states.map((s) => (
        <form key={s.id} action={stateLifeAction.bind(null, record.id, s.id, back)}>
          <Button type="submit" size="sm" variant={record.state === s.id ? "default" : "outline"} aria-pressed={record.state === s.id}>{s.label}</Button>
        </form>
      ))}
    </div>
  );
}

/** What was done with a record: a form to keep an act, and those kept, newest first. */
export function Acts({ kind, record, acts, back }: { kind: Kind; record: LifeRecord; acts: LifeAct[]; back: string }) {
  if (!kind.acts?.length) return null;
  const minutes = kind.acts.some((a) => a.minutes);
  const amount = kind.acts.some((a) => a.amount);
  const label = (id: string) => kind.acts!.find((a) => a.id === id)?.label ?? id;
  return (
    <section aria-labelledby="acts" className="grid gap-3">
      <h2 id="acts" className="text-lg font-semibold">What you did</h2>
      <form action={actLifeAction.bind(null, record.id)} className="flex flex-wrap items-end gap-2">
        <input type="hidden" name="$back" value={back} />
        <label className="grid gap-1 text-sm">What
          <select name="kind" className={cn(SELECT, "w-44")}>{kind.acts.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}</select>
        </label>
        <label className="grid gap-1 text-sm">Day<Input type="date" name="day" defaultValue={today()} className="w-40" required /></label>
        {minutes ? <label className="grid gap-1 text-sm">Minutes<Input name="minutes" inputMode="numeric" className="w-24" /></label> : null}
        {amount ? <label className="grid gap-1 text-sm">Amount<Input name="amount" inputMode="decimal" placeholder="$" className="w-28" /></label> : null}
        <label className="grid min-w-48 flex-1 gap-1 text-sm">Note<Input name="note" /></label>
        <Button type="submit" variant="outline">Keep</Button>
      </form>
      {acts.length ? (
        <ul className="divide-y rounded-md border text-sm">
          {acts.map((a) => (
            <li key={a.id} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2">
              <span className="text-muted-foreground w-24 shrink-0 tabular-nums">{a.day}</span>
              <span className="font-medium">{label(a.kind)}</span>
              {a.minutes ? <span>{a.minutes} min</span> : null}
              {a.amount !== null ? <span>${a.amount.toLocaleString("en-US")}</span> : null}
              {a.note ? <span className="text-muted-foreground min-w-0 flex-1">{a.note}</span> : null}
            </li>
          ))}
        </ul>
      ) : <p className="text-muted-foreground text-sm">Nothing kept yet.</p>}
    </section>
  );
}

/** A record in full: its state, its fields to change, what you did, and links to write about it or remove it. */
export function RecordPanel({ app, kind, record, acts, back }: { app: LifeAppId; kind: Kind; record: LifeRecord; acts: LifeAct[]; back: string }) {
  const about = `${kind.label}: ${titleOf(kind, record.body)}`;
  return (
    <div className="grid gap-8">
      <States kind={kind} record={record} back={back} />
      <RecordForm app={app} kind={kind} record={record} back={back} />
      <Acts kind={kind} record={record} acts={acts} back={back} />
      <div className="flex flex-wrap gap-2 border-t pt-4">
        <Button asChild variant="outline" size="sm"><Link href={`/journal/new?about=${encodeURIComponent(about)}`}><PenLineIcon />Write about it</Link></Button>
        <form action={removeLifeAction.bind(null, record.id)}><Button type="submit" variant="ghost" size="sm"><Trash2Icon />Remove</Button></form>
      </div>
    </div>
  );
}

/** A record as a card in a list: its name, state and the first few things it says. */
export function RecordCard({ app, kind, record, last }: { app: LifeAppId; kind: Kind; record: LifeRecord; last?: LifeAct }) {
  const state = kind.states?.find((s) => s.id === record.state);
  const facts = kind.fields.filter((f) => f.key !== kind.title && f.kind !== "long").map((f) => [f, shown(f, record.body)] as const).filter(([, v]) => v).slice(0, 3);
  return (
    <li>
      <Link href={`/life/${app}/${record.id}`} className="hover:bg-accent bg-card flex h-full flex-col gap-2 rounded-xl border p-4 shadow-xs transition">
        <span className="flex items-start justify-between gap-2">
          <span className="font-medium">{titleOf(kind, record.body)}</span>
          {state ? <span className={cn("shrink-0 rounded-full border px-2 py-0.5 text-xs", state.active ? "bg-foreground text-background" : "text-muted-foreground")}>{state.label}</span> : null}
        </span>
        {facts.length ? <span className="text-muted-foreground text-sm">{facts.map(([f, v]) => `${f.label}: ${v}`).join(" · ")}</span> : null}
        {last ? <span className="text-muted-foreground mt-auto text-xs">Last: {kind.acts?.find((a) => a.id === last.kind)?.label} on {last.day}</span> : null}
      </Link>
    </li>
  );
}
