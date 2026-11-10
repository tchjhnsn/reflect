import { SALIENCE_PROMPT } from "@/core/salience-prompt";
import { PRIMES, parseSalience, type Salience } from "@/core/state";

/** A fixed reading for tests, so no model is called. */
function stub(text: string): Salience {
  const weights = Object.fromEntries(PRIMES.map((p) => [p, 0.2]));
  weights.emotion = 0.8;
  weights.attachment = 0.6;
  const quote = "the independent one";
  return { weights, selfTerms: text.includes(quote) ? [{ quote, term: "independent", whose: "self", basis: "will" }] : [] };
}

/**
 * Ask gpt-oss on Ollama Cloud how prominent each element is in the person's words. The answer is a
 * guess the person corrects; it is never stored as a finding. Returns null when no model is set up
 * or the call fails, and reflecting goes on with the person's own labels.
 */
export async function readSalience(text: string): Promise<Salience | null> {
  if (process.env.REFLECT_READER_STUB === "1") return stub(text);
  const key = process.env.OLLAMA_API_KEY;
  if (!key) return null;
  const host = process.env.OLLAMA_HOST ?? "https://ollama.com";
  try {
    const res = await fetch(`${host}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({
        model: process.env.REFLECT_READER_MODEL ?? "gpt-oss:120b",
        stream: false,
        think: false,
        format: "json",
        options: { temperature: 0 },
        messages: [
          { role: "system", content: SALIENCE_PROMPT },
          { role: "user", content: text },
        ],
      }),
      signal: AbortSignal.timeout(45_000),
    });
    if (!res.ok) return null;
    const content = ((await res.json()) as { message?: { content?: string } }).message?.content ?? "";
    return parseSalience(text, JSON.parse(content)).salience;
  } catch {
    return null;
  }
}
