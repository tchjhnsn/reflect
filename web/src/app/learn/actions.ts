"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { BOOK_STATUS, type BookStatus } from "@reflect/core/learning";
import { lookupIsbn } from "@reflect/core/openlibrary";
import { requireUser } from "@/server/session";
import { addBook, addSubjects, keepRecall, rateSubject, setBookStatus, getBook } from "@/server/learn-store";

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
