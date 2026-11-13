"use client";

import { useState, useTransition } from "react";
import { PlusIcon, XIcon } from "lucide-react";
import { BELIEF_KINDS, DOMAINS, type BeliefKind, type Domain } from "@reflect/core/book";
import { ORIGINS, type Origin } from "@reflect/core/origins";
import type { Holder } from "@reflect/core/claims";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { beliefFromWorkAction, markBeliefAction } from "./actions";

export type IdealOption = { id: string; text: string; domain: Domain | null };

const NEW = "new";
const select = "border-input bg-background h-9 w-full rounded-md border px-2 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50";

/**
 * Mark a belief an ideal or a practical belief (after Aristotle: what is best as one would pray for it, and
 * what is best for me now). A practical belief names the ideal it serves and the circumstances it is for.
 * Either keeps its origins (the seven causes) and, for a passage, who else holds it.
 */
export function Mark({ target, domain: start, ideals, onDone, onCancel }: {
  target: { workId: string; passage: string } | { claimId: string };
  domain: Domain | null;
  ideals: IdealOption[];
  onDone: (domain: Domain | null) => void;
  onCancel: () => void;
}) {
  const fromWork = "passage" in target;
  const [kind, setKind] = useState<BeliefKind | null>(null);
  const [domain, setDomain] = useState<Domain | null>(start);
  const [serves, setServes] = useState<string>("");
  const [newIdeal, setNewIdeal] = useState("");
  const [circumstances, setCircumstances] = useState("");
  const [origins, setOrigins] = useState<Origin[]>([]);
  const [holders, setHolders] = useState<Holder[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, run] = useTransition();

  const choices = [...ideals].sort((a, b) => Number(b.domain === domain) - Number(a.domain === domain));
  const keep = () => run(async () => {
    if (!kind) return setError("Mark it as an ideal or a practical belief.");
    const m = {
      kind,
      serves: kind === "practical" ? (serves === NEW ? { text: newIdeal } : serves ? { id: serves } : null) : null,
      circumstances: kind === "practical" ? circumstances : null,
      origins,
    };
    const r = fromWork
      ? await beliefFromWorkAction(target.workId, target.passage, domain ?? "", { ...m, holders: holders.filter((h) => h.name.trim()) })
      : await markBeliefAction(target.claimId, m);
    if ("error" in r) setError(r.error);
    else onDone(domain);
  });

  return (
    <div className="space-y-4 text-sm">
      <div className="grid gap-2 sm:grid-cols-2">
        {BELIEF_KINDS.map((k) => (
          <button key={k.id} type="button" aria-pressed={kind === k.id} onClick={() => setKind(k.id)}
            className={cn("rounded-lg border px-3 py-2 text-left transition-colors", kind === k.id ? "border-foreground bg-muted" : "hover:border-foreground/30")}>
            <span className="font-medium">{k.label}</span>
            <span className="text-muted-foreground block text-xs">{k.about}</span>
          </button>
        ))}
      </div>

      {fromWork ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-muted-foreground mr-1">Domain</span>
          {DOMAINS.map((d) => <Button key={d.id} type="button" size="sm" variant={domain === d.id ? "default" : "outline"} className="h-7" aria-pressed={domain === d.id} onClick={() => setDomain(d.id)}>{d.title}</Button>)}
        </div>
      ) : null}

      {kind === "practical" ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1">
            <span className="text-muted-foreground">The ideal it serves</span>
            <select className={select} value={serves} onChange={(e) => setServes(e.target.value)}>
              <option value="">Choose an ideal</option>
              {choices.map((i) => <option key={i.id} value={i.id}>{i.text}</option>)}
              <option value={NEW}>A new ideal...</option>
            </select>
            {serves === NEW ? <Input autoFocus value={newIdeal} onChange={(e) => setNewIdeal(e.target.value)} placeholder="What is best, simply" /> : null}
          </label>
          <label className="space-y-1">
            <span className="text-muted-foreground">The circumstances it is for</span>
            <Input value={circumstances} onChange={(e) => setCircumstances(e.target.value)} placeholder="While the children are small" />
          </label>
        </div>
      ) : null}

      {kind ? (
        <div className="space-y-1.5">
          <span className="text-muted-foreground">Where it comes from</span>
          <div className="flex flex-wrap items-center gap-1.5">
            {ORIGINS.map((o, i) => (
              <span key={o.id} className="contents">
                {i === 3 ? <span className="bg-border mx-1 h-5 w-px" aria-hidden="true" /> : null}
                <Button type="button" size="sm" variant={origins.includes(o.id) ? "default" : "outline"} className="h-7" aria-pressed={origins.includes(o.id)}
                  onClick={() => setOrigins(origins.includes(o.id) ? origins.filter((x) => x !== o.id) : [...origins, o.id])}>{o.label}</Button>
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {kind && fromWork ? (
        <div className="space-y-1.5">
          <span className="text-muted-foreground">Who else holds it</span>
          {holders.map((h, i) => (
            <div key={i} className="flex gap-2">
              <Input value={h.name} placeholder="Name" aria-label="Holder" onChange={(e) => setHolders(holders.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} />
              <Input value={h.citation ?? ""} placeholder="Where they say it" aria-label="Citation" onChange={(e) => setHolders(holders.map((x, j) => (j === i ? { ...x, citation: e.target.value } : x)))} />
              <Button type="button" size="icon" variant="ghost" aria-label="Remove" onClick={() => setHolders(holders.filter((_, j) => j !== i))}><XIcon /></Button>
            </div>
          ))}
          <Button type="button" size="sm" variant="ghost" className="h-7" onClick={() => setHolders([...holders, { name: "" }])}><PlusIcon />Add a holder</Button>
        </div>
      ) : null}

      <div className="flex items-center gap-2">
        <Button type="button" size="sm" disabled={pending || !kind || (fromWork && !domain)} onClick={keep}>Keep</Button>
        <Button type="button" size="sm" variant="ghost" onClick={onCancel}>Cancel</Button>
        {error ? <span className="text-destructive" role="alert">{error}</span> : null}
      </div>
    </div>
  );
}
