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
];

export const listById = (id: string): BookList | undefined => LISTS.find((l) => l.id === id);

/** The open library search that finds one book of a list: its title, and its author when the list names one. */
export const listBookQuery = (b: ListBook): string => `title:"${b.title.replace(/"/g, "")}"${b.author ? ` author:"${b.author.replace(/"/g, "")}"` : ""}`;
