import { SALIENCE_PROMPT } from "./salience-prompt.ts";
import { PRIMES, parseSalience, type Salience } from "./state.ts";

// The reader proposes; the person decides. Jev (a TypeSafe System 1 model) answers a yes/no question
// per element and picks the basis of each self-term, with probabilities. gpt-oss only finds the
// self-terms, because Jev answers questions and does not quote. Nothing here is stored as a finding.

/** What each element looks like in first-person writing, worded for a yes/no question. */
const LOOKS_LIKE: Record<string, string> = {
  will: "choosing, asserting or refusing",
  desire: "wanting or longing for something",
  imagination: "picturing what is not present",
  logos: "putting things into words, names or symbols",
  emotion: "the writer's own feelings or mood",
  intuition: "knowing without reasoning, a gut sense",
  conscience: "what the writer ought to do, right and wrong",
  reason: "working things out with logic or evidence",
  attention: "what the writer's focus is on",
  awareness: "the writer noticing their own experience",
  "self-differentiation": "the writer defining or comparing themselves",
  attachment: "the writer's bond to a particular person",
  embodiment: "the body, its state, or physical place",
  temporality: "time, duration, before and after",
  "aesthetic-sense": "beauty or fittingness",
  "numinous-sense": "the sacred, awe or transcendence",
};

/** Jev's yes/no answers run high, so a guess is shown only when it is near certain. */
export const JEV_GUESS_AT = 0.9;

const key = (p: string) => p.replaceAll("-", "_");
const BASES = PRIMES.filter((p) => p !== "self-differentiation");

/** A fixed reading for tests, so no model is called. */
function stub(text: string): Salience {
  const weights = Object.fromEntries(PRIMES.map((p) => [p, 0.2]));
  weights.emotion = 0.95;
  weights.attachment = 0.92;
  const quote = "the independent one";
  const odds = { imagination: 0.52, will: 0.28, reason: 0.11, emotion: 0.09 };
  return {
    weights,
    guessAt: JEV_GUESS_AT,
    selfTerms: text.includes(quote) ? [{ quote, term: "independent", whose: "self", basis: "imagination", odds }] : [],
  };
}

export type Jev = { url: string; model: string; token: string };

/**
 * Jev through Vercel AI Gateway when there is a gateway token (`AI_GATEWAY_API_KEY`, or the token the
 * caller passes, such as a Vercel deployment's OIDC token), else TypeSafe directly (`TYPESAFEAI_API_KEY`).
 */
export function jevEndpoint(gatewayToken?: string): Jev | null {
  const token = process.env.AI_GATEWAY_API_KEY || gatewayToken;
  if (token) return { url: "https://ai-gateway.vercel.sh/typesafe/v1/systemone", model: "typesafe-ai/jev", token };
  const direct = process.env.TYPESAFEAI_API_KEY;
  return direct ? { url: "https://api.typesafe.ai/v1/systemone", model: "jev-latest", token: direct } : null;
}

type Question = { type: "noul"; instructions: string } | { type: "choice"; instructions: string; criteria: Record<string, string> };
type Answer = { type: "noul"; noul: number } | { type: "choice"; choice: string; probabilities?: Record<string, number> };

async function askJev(jev: Jev, state: string, questions: Record<string, Question>): Promise<Record<string, Answer>> {
  const res = await fetch(jev.url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${jev.token}` },
    body: JSON.stringify({ model: jev.model, state, questions }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`jev ${res.status}`);
  return ((await res.json()) as { answers: Record<string, Answer> }).answers;
}

const FIND_TERMS = `You find comparative terms a writer assigns to themselves or to another person ("tall", "the quiet one", "a late bloomer", "the plain sister").
Return JSON: {"self_terms":[{"quote":"exact words from the text","term":"the term in a word or two","whose":"self"|"other"}]}.
The quote is the term as written, copied exactly, not the people it compares with. Return {"self_terms":[]} when there are none. Do not judge or explain.`;

type Found = { quote: string; term: string; whose: "self" | "other" };

/** gpt-oss through AI Gateway when it is set up, else Ollama Cloud. Returns null when neither is. */
export async function chatJson(system: string, text: string, gateway: string | null): Promise<unknown | null> {
  const messages = [
    { role: "system", content: system },
    { role: "user", content: text },
  ];
  if (gateway) {
    const res = await fetch("https://ai-gateway.vercel.sh/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${gateway}` },
      body: JSON.stringify({ model: "openai/gpt-oss-120b", messages, temperature: 0, response_format: { type: "json_object" } }),
      signal: AbortSignal.timeout(45_000),
    });
    if (!res.ok) return null;
    const content = ((await res.json()) as { choices?: { message?: { content?: string } }[] }).choices?.[0]?.message?.content ?? "";
    return JSON.parse(content);
  }
  const ollama = process.env.OLLAMA_API_KEY;
  if (!ollama) return null;
  const res = await fetch(`${process.env.OLLAMA_HOST ?? "https://ollama.com"}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${ollama}` },
    body: JSON.stringify({ model: process.env.REFLECT_READER_MODEL ?? "gpt-oss:120b", stream: false, think: false, format: "json", options: { temperature: 0 }, messages }),
    signal: AbortSignal.timeout(45_000),
  });
  if (!res.ok) return null;
  return JSON.parse(((await res.json()) as { message?: { content?: string } }).message?.content ?? "");
}

/** Keep only terms whose quote is really in the text. */
export function foundTerms(text: string, answer: unknown): Found[] {
  const list = (answer as { self_terms?: unknown } | null)?.self_terms;
  const out: Found[] = [];
  for (const t of Array.isArray(list) ? list : []) {
    const { quote, term, whose } = (t ?? {}) as Record<string, unknown>;
    if (typeof quote !== "string" || !quote.trim() || !text.includes(quote)) continue;
    if (whose !== "self" && whose !== "other") continue;
    if (out.some((o) => o.quote === quote)) continue;
    out.push({ quote, term: typeof term === "string" && term ? term : quote, whose });
  }
  return out.slice(0, 5);
}

/** Jev's answers as a reading: a weight per element, and each term's basis with its odds. */
export function fromJev(answers: Record<string, Answer>, terms: Found[]): Salience | null {
  const weights: Record<string, number> = {};
  for (const p of PRIMES) {
    const a = answers[key(p)];
    if (a?.type !== "noul" || !(a.noul >= 0 && a.noul <= 1)) return null;
    weights[p] = a.noul;
  }
  const selfTerms: Salience["selfTerms"] = [];
  terms.forEach((t, i) => {
    const a = answers[`basis_${i}`];
    if (a?.type !== "choice") return;
    const basis = BASES.find((p) => key(p) === a.choice);
    if (!basis) return;
    const odds: Record<string, number> = {};
    for (const p of BASES) {
      const v = a.probabilities?.[key(p)];
      if (typeof v === "number" && v >= 0 && v <= 1) odds[p] = v;
    }
    selfTerms.push({ ...t, basis, odds });
  });
  return { weights, guessAt: JEV_GUESS_AT, selfTerms };
}

function basisQuestion(t: Found): Question {
  const who = t.whose === "self" ? "The writer calls themselves" : "The writer calls someone";
  return {
    type: "choice",
    instructions: `${who} "${t.quote}". That is a comparison. Which faculty is the comparison made through?`,
    criteria: Object.fromEntries(BASES.map((p) => [key(p), LOOKS_LIKE[p]!])),
  };
}

/** The older path: gpt-oss alone gives weights and bases. Used when Jev is not set up. */
async function readWithGptOss(text: string): Promise<Salience | null> {
  const answer = await chatJson(SALIENCE_PROMPT, text, null);
  return answer === null ? null : parseSalience(text, answer).salience;
}

/**
 * Propose how prominent each element is in the person's words, and the basis of each self-term.
 * Returns null when no model is set up or a call fails; reflecting goes on with the person's own labels.
 */
export async function readSalience(text: string, jev: Jev | null = jevEndpoint()): Promise<Salience | null> {
  if (process.env.REFLECT_READER_STUB === "1") return stub(text);
  try {
    if (!jev) return await readWithGptOss(text);
    const gateway = jev.url.startsWith("https://ai-gateway") ? jev.token : null;
    const elements: Record<string, Question> = Object.fromEntries(
      PRIMES.map((p) => [key(p), { type: "noul", instructions: `Is ${LOOKS_LIKE[p]} prominent in this first-person writing?` }]),
    );
    const [answers, terms] = await Promise.all([
      askJev(jev, text, elements),
      chatJson(FIND_TERMS, text, gateway).then((a) => foundTerms(text, a)).catch(() => []),
    ]);
    if (terms.length) Object.assign(answers, await askJev(jev, text, Object.fromEntries(terms.map((t, i) => [`basis_${i}`, basisQuestion(t)]))).catch(() => ({})));
    return fromJev(answers, terms);
  } catch {
    return null;
  }
}
