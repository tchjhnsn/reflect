"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { BOOK_STATUS, FILM_STATUS, type BookStatus, type FilmStatus } from "@reflect/core/learning";
import { readSweep } from "@reflect/core/sweep";
import { addInput } from "@/server/ideas-store";
import { keepSweepItems } from "@/server/sweep-store";
import { lookupIsbn } from "@reflect/core/openlibrary";
import { requireUser } from "@/server/session";
import { addBook, addSubjects, keepRecall, rateSubject, setBookStatus, getBook } from "@/server/learn-store";
import { addCredential, getUpload, keepCourses, keepLinkedIn, listCourses, saveUpload } from "@/server/learn-sources-store";
import { checkBadge } from "@reflect/core/credentials";
import { readLinkedInExport } from "@reflect/core/linkedin";
import { draftTranscript, transcriptText } from "@reflect/core/transcript";
import { chatJson } from "@reflect/core/chat";
import { getVercelOidcToken } from "@vercel/oidc";
import { unzipSync } from "fflate";

const list = (v: FormDataEntryValue | null) => String(v ?? "").split(",").map((x) => x.trim()).filter(Boolean);
const status = (v: FormDataEntryValue | null): BookStatus => ((BOOK_STATUS as readonly string[]).includes(String(v)) ? (String(v) as BookStatus) : "want");
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });

/** Keep a book. With an ISBN, Open Library fills the authors and its subjects are offered, not kept. */
export async function addBookAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const isbn = String(form.get("isbn") ?? "").trim();
  const catalog = isbn ? await lookupIsbn(isbn, openLibrary).catch(() => null) : null;
  const id = await addBook(user.id, { title: String(form.get("title") ?? ""), authors: catalog?.authors ?? [], ...(isbn ? { isbn } : {}), status: status(form.get("status")), subjects: list(form.get("subjects")), catalog });
  revalidatePath("/learn");
  redirect(`/learn/book/${id}`);
}

export async function setStatusAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("book") ?? "");
  await setBookStatus(user.id, id, status(form.get("status")));
  revalidatePath(`/learn/book/${id}`);
}

export async function addSubjectsAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("book") ?? "");
  await addSubjects(user.id, id, [...list(form.get("subjects")), ...form.getAll("suggested").map(String)]);
  revalidatePath(`/learn/book/${id}`);
}

export async function recallAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("book") ?? "");
  await keepRecall(user.id, id, String(form.get("recall") ?? ""));
  revalidatePath(`/learn/book/${id}`);
}

/** The person's grasp of each subject they rated, from this book. Subjects left unrated stay unrated. */
export async function rateAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("book") ?? "");
  const book = await getBook(user.id, id);
  if (!book) throw new Error("no such book");
  for (const s of book.subjects) {
    const g = form.get(`grasp-${s.id}`);
    if (g) await rateSubject(user.id, s.name, String(g), { bookId: id });
  }
  revalidatePath(`/learn/book/${id}`);
  revalidatePath("/learn");
}

async function gateway(): Promise<string | null> {
  if (process.env.AI_GATEWAY_API_KEY) return process.env.AI_GATEWAY_API_KEY;
  return process.env.VERCEL ? ((await getVercelOidcToken().catch(() => undefined)) ?? null) : null;
}

/** Upload a transcript: its text is read, a model drafts its rows, and nothing is kept as a course yet. */
export async function uploadTranscriptAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const file = form.get("file");
  if (!(file instanceof File) || !file.size) throw new Error("choose your transcript, a PDF or a text file");
  const bytes = new Uint8Array(await file.arrayBuffer());
  const text = /\.pdf$/i.test(file.name) ? (await (await import("unpdf")).extractText(bytes.slice(), { mergePages: true })).text : transcriptText(bytes, file.name);
  const token = await gateway();
  const draft = await draftTranscript(text, (system, t) => chatJson(system, t, token));
  if (!draft) throw new Error("reading transcripts is not set up here yet");
  const id = await saveUpload(user.id, "transcript", file.name, draft, bytes);
  redirect(`/learn/upload/${id}`);
}

export async function keepCoursesAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("upload") ?? "");
  await keepCourses(user.id, id, form.getAll("row").map(Number).filter((n) => Number.isInteger(n) && n >= 0));
  revalidatePath(`/learn/upload/${id}`);
  revalidatePath("/learn");
}

/** The person's grasp of courses, by the subject each is linked to. */
export async function rateCoursesAction(form: FormData): Promise<void> {
  const user = await requireUser();
  for (const c of await listCourses(user.id)) {
    const g = form.get(`grasp-${c.id}`);
    if (g) await rateSubject(user.id, c.subject.name, String(g));
  }
  revalidatePath("/learn");
}

export async function addCredentialAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const url = String(form.get("url") ?? "").trim();
  const checked = url && form.get("check") === "on" ? await checkBadge(url, openLibrary) : null;
  const opt = (k: string) => String(form.get(k) ?? "").trim();
  await addCredential(user.id, opt("name") || checked?.name || "", { issuer: opt("issuer") || checked?.issuer || "", ...(opt("issued") || checked?.issued ? { issued: opt("issued") || checked!.issued } : {}), ...(url ? { url } : {}), subjects: list(form.get("subjects")), ...(checked ? { checked: { ...checked, at: new Date().toISOString() } } : {}) });
  revalidatePath("/learn");
}

/** Upload a LinkedIn export: what it holds is shown first, and kept only when the person says. */
export async function uploadLinkedInAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const file = form.get("file");
  if (!(file instanceof File) || !file.size) throw new Error("choose the zip LinkedIn sent you");
  if (file.size > 50_000_000) throw new Error("the export is larger than 50 MB");
  const files = new Map<string, string>();
  for (const [k, v] of Object.entries(unzipSync(new Uint8Array(await file.arrayBuffer()), { filter: (f) => f.name.toLowerCase().endsWith(".csv") }))) files.set(k, new TextDecoder().decode(v));
  const id = await saveUpload(user.id, "linkedin", file.name, readLinkedInExport(files));
  redirect(`/learn/upload/${id}`);
}

export async function keepLinkedInAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("upload") ?? "");
  const u = await getUpload(user.id, id);
  if (!u?.linkedin) throw new Error("no such export");
  const on = (k: string) => form.get(k) === "on";
  await keepLinkedIn(user.id, u.linkedin, { endorsements: on("endorsements"), recommendations: on("recommendations"), certifications: on("certifications"), skills: on("skills") });
  revalidatePath("/learn");
  redirect("/learn");
}

const opt = (form: FormData, k: string) => String(form.get(k) ?? "").trim();

/** Keep a film, talk, conversation or link: an input beside the books. */
export async function addInputAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const kind = opt(form, "kind");
  const title = opt(form, "title");
  const url = opt(form, "url");
  const on = opt(form, "on");
  const subjects = list(form.get("subjects"));
  if (kind === "film") await addInput(user.id, { kind: "film", title, film: opt(form, "film") || "film", ...(opt(form, "year") ? { year: Number(opt(form, "year")) } : {}), status: ((FILM_STATUS as readonly string[]).includes(opt(form, "status")) ? opt(form, "status") : "watched") as FilmStatus }, subjects);
  else if (kind === "talk") await addInput(user.id, { kind: "talk", title, speaker: opt(form, "who"), ...(on ? { on } : {}), ...(url ? { url } : {}) }, subjects);
  else if (kind === "conversation") await addInput(user.id, { kind: "conversation", with: opt(form, "who").split(/[;,]/), on: on || new Date().toISOString().slice(0, 10), about: title }, subjects);
  else if (kind === "link") await addInput(user.id, { kind: "link", title: title || url, link: opt(form, "link") || "article", ...(url ? { url } : {}), ...(opt(form, "who") ? { by: opt(form, "who") } : {}) }, subjects);
  else throw new Error("choose a film, a talk, a conversation or a link");
  revalidatePath("/learn");
}

/** Read the reply of the person's own AI to the memory sweep prompt; nothing is kept until they choose. */
export async function sweepAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const file = form.get("file");
  const text = file instanceof File && file.size ? await file.text() : opt(form, "reply");
  if (!text) throw new Error("paste your AI's reply, or choose the file it is in");
  if (text.length > 2_000_000) throw new Error("the reply is too long");
  const id = await saveUpload(user.id, "sweep", file instanceof File && file.size ? file.name : "pasted reply", readSweep(text));
  redirect(`/learn/upload/${id}`);
}

export async function keepSweepAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = String(form.get("upload") ?? "");
  await keepSweepItems(user.id, id, form.getAll("row").map(Number).filter((n) => Number.isInteger(n) && n >= 0));
  revalidatePath(`/learn/upload/${id}`);
  revalidatePath("/learn");
  revalidatePath("/ideas");
}
