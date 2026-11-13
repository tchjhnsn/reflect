// Lists of books chosen by others, to browse as a set. Each is a public, dated selection with its source.
// A list names its books by title and author; the open library finds each one when the list is shown.
//
// The Harvard Classics: the fifty-volume "Five-Foot Shelf of Books" chosen by Charles W. Eliot, president of
// Harvard, published by P. F. Collier & Son in 1909 and 1910 (in the public domain). Listed here are the
// principal works of its volumes, in volume order. Source: Eliot (ed.), The Harvard Classics, vols. 1-50.

export type ListBook = { title: string; author: string };
export type BookList = { id: string; name: string; about: string; source: string; books: ListBook[] };

export const LISTS: readonly BookList[] = [
  {
    id: "harvard-classics",
    name: "The Harvard Classics",
    about: "The \"Five-Foot Shelf of Books\" that Charles W. Eliot, president of Harvard, chose in 1909 as a liberal education in reading: the principal works of its fifty volumes, in volume order.",
    source: "Charles W. Eliot (ed.), The Harvard Classics, P. F. Collier & Son, 1909-1910",
    books: [
      { title: "The Autobiography of Benjamin Franklin", author: "Benjamin Franklin" },
      { title: "The Journal of John Woolman", author: "John Woolman" },
      { title: "Apology", author: "Plato" },
      { title: "Phaedo", author: "Plato" },
      { title: "Meditations", author: "Marcus Aurelius" },
      { title: "Essays", author: "Francis Bacon" },
      { title: "Areopagitica", author: "John Milton" },
      { title: "Religio Medici", author: "Thomas Browne" },
      { title: "Paradise Lost", author: "John Milton" },
      { title: "Essays", author: "Ralph Waldo Emerson" },
      { title: "Confessions", author: "Augustine of Hippo" },
      { title: "The Imitation of Christ", author: "Thomas à Kempis" },
      { title: "Agamemnon", author: "Aeschylus" },
      { title: "Oedipus the King", author: "Sophocles" },
      { title: "On Friendship", author: "Cicero" },
      { title: "The Wealth of Nations", author: "Adam Smith" },
      { title: "On the Origin of Species", author: "Charles Darwin" },
      { title: "Lives", author: "Plutarch" },
      { title: "Aeneid", author: "Virgil" },
      { title: "Don Quixote", author: "Miguel de Cervantes" },
      { title: "The Pilgrim's Progress", author: "John Bunyan" },
      { title: "The Arabian Nights", author: "" },
      { title: "Fables", author: "Aesop" },
      { title: "Faust", author: "Johann Wolfgang von Goethe" },
      { title: "Doctor Faustus", author: "Christopher Marlowe" },
      { title: "The Divine Comedy", author: "Dante Alighieri" },
      { title: "The Betrothed", author: "Alessandro Manzoni" },
      { title: "The Odyssey", author: "Homer" },
      { title: "Two Years Before the Mast", author: "Richard Henry Dana" },
      { title: "Reflections on the Revolution in France", author: "Edmund Burke" },
      { title: "On Liberty", author: "John Stuart Mill" },
      { title: "The Voyage of the Beagle", author: "Charles Darwin" },
      { title: "The Autobiography of Benvenuto Cellini", author: "Benvenuto Cellini" },
      { title: "Discourse on Method", author: "René Descartes" },
      { title: "The Prince", author: "Niccolò Machiavelli" },
      { title: "Utopia", author: "Thomas More" },
      { title: "An Essay Concerning Human Understanding", author: "John Locke" },
      { title: "An Enquiry Concerning Human Understanding", author: "David Hume" },
      { title: "Pensées", author: "Blaise Pascal" },
      { title: "Beowulf", author: "" },
      { title: "The Song of Roland", author: "" },
    ],
  },
  /**
   * The Last Days of Socrates: the four dialogues of Socrates' trial and death, in the order of the events they tell
   * (the order of Hugh Tredennick's Penguin translation, 1954, under that title). Plato wrote them; they are in the
   * public domain in Benjamin Jowett's translation.
   */
  {
    id: "last-days-of-socrates",
    name: "The Last Days of Socrates",
    about: "Plato's four dialogues of Socrates' trial and death, read in the order of the events: on his way to the court, his defence, in prison, and his last day.",
    source: "Plato, Euthyphro, Apology, Crito, Phaedo; read together as The Last Days of Socrates (tr. Hugh Tredennick, Penguin, 1954)",
    books: [
      { title: "Euthyphro", author: "Plato" },
      { title: "Apology", author: "Plato" },
      { title: "Crito", author: "Plato" },
      { title: "Phaedo", author: "Plato" },
    ],
  },
];

/**
 * Where a person is on a list read in order: each book's status (from their shelves, or null when not there), how
 * many are read, and the first book not yet read.
 */
export function pathProgress<S extends string>(statuses: readonly (S | null)[]): { read: number; next: number | null } {
  const read = statuses.filter((s) => s === "read").length;
  const i = statuses.findIndex((s) => s !== "read");
  return { read, next: i < 0 ? null : i };
}

export const listById = (id: string): BookList | undefined => LISTS.find((l) => l.id === id);

/** The open library search that finds one book of a list: its title, and its author when the list names one. */
/**
 * Which search result is the list's book itself. A title search also finds collections that hold it ("Euthyphro,
 * Apology, Crito"), often ranked first; the result whose title is the book's own title, give or take an article or
 * the author's name ("Plato's Phaedo", "Platonis Crito"), wins, then the first that is not a collection, then the
 * first result.
 */
export function bestListMatch<T extends { title: string }>(b: ListBook, results: T[]): T | null {
  const author = b.author.toLowerCase().split(/\s+/).filter(Boolean);
  const skip = new Set(["the", "a", "an", "of", ...author, ...author.flatMap((w) => [`${w}s`, `${w}'s`, `${w}nis`, `${w.replace(/o$/, "")}onis`])]);
  const words = (t: string) => t.toLowerCase().replace(/[’']s\b/g, "s").replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) => w && !skip.has(w));
  const want = words(b.title).join(" ");
  const exact = results.find((r) => words(r.title).join(" ") === want);
  if (exact) return exact;
  const single = results.find((r) => !/[,;&/]| and /i.test(r.title) && words(r.title).join(" ").includes(want));
  return single ?? results[0] ?? null;
}

export const listBookQuery = (b: ListBook): string => `title:"${b.title.replace(/"/g, "")}"${b.author ? ` author:"${b.author.replace(/"/g, "")}"` : ""}`;
