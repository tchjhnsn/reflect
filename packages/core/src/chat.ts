// A JSON answer from gpt-oss, through Vercel AI Gateway when the caller has a gateway token, else Ollama
// Cloud. Reflect uses it to draft (a transcript's courses), never to decide: the person keeps what fits.
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
