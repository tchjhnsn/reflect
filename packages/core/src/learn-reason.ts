// Why a person is learning something (Torian, 2026-11-14), asked before a course: what is causing them to learn it
// (the moving cause, marked with the seven origins of Rhetoric I.10) and for the sake of what (the end, Physics
// II.3), for its own sake or for a use (Metaphysics 982b24; Politics VIII.2 1337b). The writing is a work of the
// journal under two headings; the origins and the kind of end are kept beside it.

export const REASON = { cause: 4000, end: 4000 } as const;
export const REASON_HEADINGS = { cause: "What is causing me to learn this", end: "For the sake of what" } as const;

export type EndKind = "own-sake" | "use";
export const END_KINDS: { id: EndKind; label: string }[] = [
  { id: "own-sake", label: "For its own sake" },
  { id: "use", label: "For a use" },
];
export const isEndKind = (v: unknown): v is EndKind => v === "own-sake" || v === "use";

const clip = (v: unknown, max: number) => (typeof v === "string" ? v.replace(/\r\n?/g, "\n").trim().slice(0, max) : "");

/** The work's text: each answer under its heading. */
export function reasonBody(cause: unknown, end: unknown): string {
  return `## ${REASON_HEADINGS.cause}\n\n${clip(cause, REASON.cause)}\n\n## ${REASON_HEADINGS.end}\n\n${clip(end, REASON.end)}\n`;
}

/** The two answers read back from the work's text; text outside the headings (edited in the journal) goes with the cause. */
export function reasonParts(body: string): { cause: string; end: string } {
  const at = (h: string) => body.indexOf(`## ${h}`);
  const c = at(REASON_HEADINGS.cause);
  const e = at(REASON_HEADINGS.end);
  if (e < 0) return { cause: body.replace(`## ${REASON_HEADINGS.cause}`, "").trim(), end: "" };
  const before = body.slice(0, Math.max(c, 0) < e && c >= 0 ? c : e);
  const cause = c >= 0 && c < e ? body.slice(c + REASON_HEADINGS.cause.length + 3, e) : "";
  return { cause: `${before.trim()}\n\n${cause.trim()}`.trim(), end: body.slice(e + REASON_HEADINGS.end.length + 3).trim() };
}
