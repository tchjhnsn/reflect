import type { Target } from "@reflect/core/ideas";
import type { BookStatus, FilmStatus } from "@reflect/core/learning";
import type { SweepItem } from "@reflect/core/sweep";
import { db } from "@/db/pool";
import { addInput, keepIdea, markSubject, resolveByTitle } from "./ideas-store";
import { addCourses, addCredential, getUpload } from "./learn-sources-store";
import { addBook } from "./learn-store";

const BOOK: Record<string, BookStatus> = { finished: "read", started: "reading", referenced: "referenced", wants: "want" };
const FILM: Record<string, FilmStatus> = { finished: "watched", started: "watching", referenced: "referenced", wants: "want" };
const LINKS = new Set(["post", "thread", "article", "video", "newsletter", "paper", "podcast"]);
const http = (u?: string) => (u && /^https?:\/\/\S+$/.test(u) ? u : undefined);

/**
 * Keep the items of a memory sweep the person chose (0-based), once each. Ideas go last so their inputs
 * are found by title; an input the person's AI only inferred is kept as maybe from. Nothing sets a grasp.
 */
export async function keepSweepItems(userId: string, uploadId: string, rows: number[]): Promise<number> {
  const u = await getUpload(userId, uploadId);
  if (!u?.sweep) throw new Error("no such sweep");
  const s = u.sweep;
  const chosen = [...new Set(rows)].filter((i) => s.items[i] && !u.kept.includes(i));
  const order = [...chosen.filter((i) => s.items[i]!.kind !== "idea"), ...chosen.filter((i) => s.items[i]!.kind === "idea")];
  const source = (i: SweepItem) => `${s.assistant}${s.reportedAt ? ` ${s.reportedAt}` : ""}, ${i.basis === "i-said" ? "you said" : "it inferred"}${i.evidence ? `: ${i.evidence}` : ""}`;
  for (const n of order) {
    const i = s.items[n]!;
    if (i.kind === "book") await addBook(userId, { title: i.title, authors: i.authors, status: BOOK[i.engaged] ?? "referenced", subjects: i.subjects });
    else if (i.kind === "film") await addInput(userId, { kind: "film", title: i.title, film: ["film", "documentary", "series", "short"].includes(i.film) ? i.film : "film", ...(i.year ? { year: i.year } : {}), status: FILM[i.engaged] ?? "referenced" }, i.subjects, source(i));
    else if (i.kind === "talk") await addInput(userId, { kind: "talk", title: i.title, speaker: i.speaker, ...(http(i.url) ? { url: http(i.url) } : {}) }, i.subjects, source(i));
    else if (i.kind === "link") await addInput(userId, { kind: "link", title: i.title, link: LINKS.has(i.link) ? i.link : "article", ...(http(i.url) ? { url: http(i.url) } : {}), ...(i.by ? { by: i.by } : {}) }, i.subjects, source(i));
    else if (i.kind === "conversation") await addInput(userId, { kind: "conversation", with: i.with, on: i.when && /^\d{4}(-\d{2}(-\d{2})?)?$/.test(i.when) ? i.when : new Date().toISOString().slice(0, 10), about: i.about }, [], source(i));
    else if (i.kind === "course") await addCourses(userId, [{ term: "", title: i.title, ...(i.where ? { institution: i.where } : {}) }]);
    else if (i.kind === "credential") await addCredential(userId, i.name, { issuer: i.issuer });
    else if (i.kind === "shown") await markSubject(userId, i.name, "shown", s.assistant, `${i.how ? `${i.how}: ` : ""}${i.evidence ?? "it saw you show it"}`);
    else if (i.kind === "suggested") await markSubject(userId, i.name, "suggested", s.assistant, i.because);
    else {
      const inputs: { target: Target; how?: string }[] = [];
      for (const x of i.inputs) inputs.push({ target: await resolveByTitle(userId, x.title), ...(x.how ? { how: x.how } : {}) });
      await keepIdea(userId, i.title, { statement: i.statement, ...(i.basis === "i-said" ? { from: inputs } : { maybe: inputs }) });
    }
    await db().query("INSERT INTO learn_upload_kept (upload_id, row_index) VALUES ($1, $2) ON CONFLICT DO NOTHING", [uploadId, n]);
  }
  return order.length;
}
