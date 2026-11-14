/**
 * Lists beyond books, each chosen by others and credited to its source: open university courses, famous public
 * debates, the Greek tragedies Aristotle's Poetics draws on, and the films the Vatican named in 1995. A link is
 * given only where the publisher's own page was checked; the rest are named, to be found and added.
 */
import type { Medium, Way } from "./ways";
import { mediumOf } from "./ways";

export type ListItem = { title: string; by?: string; medium: Medium; url?: string; about: string[] };
export type ItemList = { id: string; name: string; about: string; source: string; items: ListItem[] };

export const ITEM_LISTS: readonly ItemList[] = [
  {
    id: "open-yale-courses",
    name: "Open Yale Courses",
    about: "Whole Yale lecture courses, recorded in class and published free with transcripts.",
    source: "Yale University, Open Yale Courses (oyc.yale.edu)",
    items: [
      { title: "Death (PHIL 176)", by: "Shelly Kagan", medium: "course", url: "https://oyc.yale.edu/philosophy/phil-176", about: ["death", "philosophy", "mortality", "soul"] },
      { title: "Introduction to Political Philosophy (PLSC 114)", by: "Steven B. Smith", medium: "course", url: "https://oyc.yale.edu/political-science/plsc-114", about: ["political philosophy", "politics", "aristotle", "plato"] },
      { title: "Introduction to Theory of Literature (ENGL 300)", by: "Paul H. Fry", medium: "course", url: "https://oyc.yale.edu/english/engl-300", about: ["literature", "literary theory", "criticism"] },
      { title: "Introduction to New Testament History and Literature (RLST 152)", by: "Dale B. Martin", medium: "course", url: "https://oyc.yale.edu/religious-studies/rlst-152", about: ["new testament", "bible", "christianity", "religion"] },
      { title: "Introduction to Ancient Greek History (CLCV 205)", by: "Donald Kagan", medium: "course", url: "https://oyc.yale.edu/classics/clcv-205", about: ["ancient greece", "greek history", "history", "athens"] },
      { title: "The Civil War and Reconstruction Era, 1845-1877 (HIST 119)", by: "David W. Blight", medium: "course", url: "https://oyc.yale.edu/history/hist-119", about: ["civil war", "american history", "slavery", "reconstruction"] },
      { title: "Fundamentals of Physics I (PHYS 200)", by: "Ramamurti Shankar", medium: "course", url: "https://oyc.yale.edu/physics/phys-200", about: ["physics", "mechanics", "science"] },
    ],
  },
  {
    id: "mit-opencourseware",
    name: "MIT OpenCourseWare",
    about: "MIT courses published free with lectures, notes and problem sets.",
    source: "Massachusetts Institute of Technology, MIT OpenCourseWare (ocw.mit.edu)",
    items: [
      { title: "Linear Algebra (18.06)", by: "Gilbert Strang", medium: "course", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", about: ["linear algebra", "mathematics", "matrices"] },
      { title: "Classical Mechanics (8.01SC)", by: "MIT Department of Physics", medium: "course", url: "https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/", about: ["physics", "mechanics", "science"] },
    ],
  },
  {
    id: "justice-harvard",
    name: "Justice, Harvard",
    about: "Michael Sandel's Harvard course on moral and political philosophy, filmed in the lecture hall.",
    source: "Harvard University and WGBH, Justice: What's the Right Thing to Do? (2009)",
    items: [
      { title: "Justice: What's the Right Thing to Do? Episode 1, The Moral Side of Murder", by: "Michael Sandel, Harvard University", medium: "lecture", url: "https://www.youtube.com/watch?v=kBdfcR-8hEY", about: ["justice", "ethics", "moral philosophy", "utilitarianism"] },
    ],
  },
  {
    id: "famous-debates",
    name: "Famous public debates",
    about: "Debates remembered for testing opinions in public, each with who argued it and when.",
    source: "The historical record of each debate (transcripts and recordings are widely published)",
    items: [
      { title: "The Lincoln-Douglas debates (1858)", by: "Abraham Lincoln and Stephen A. Douglas", medium: "debate", about: ["slavery", "american history", "politics", "union"] },
      { title: "The existence of God (BBC radio, 1948)", by: "Bertrand Russell and Frederick Copleston", medium: "debate", about: ["god", "religion", "philosophy", "metaphysics"] },
      { title: "Has the American Dream been achieved at the expense of the American Negro? (Cambridge Union, 1965)", by: "James Baldwin and William F. Buckley Jr.", medium: "debate", about: ["race", "america", "civil rights"] },
      { title: "Human Nature: Justice versus Power (1971)", by: "Noam Chomsky and Michel Foucault", medium: "debate", about: ["human nature", "justice", "power", "philosophy"] },
    ],
  },
  {
    id: "greek-tragedies",
    name: "The Greek tragedies",
    about: "The Athenian tragedies, the plays Aristotle's Poetics studies; Oedipus the King is its chief example.",
    source: "Aristotle, Poetics; the surviving plays of Aeschylus, Sophocles and Euripides",
    items: [
      { title: "Oedipus the King", by: "Sophocles", medium: "theatre", about: ["tragedy", "fate", "greek drama"] },
      { title: "Antigone", by: "Sophocles", medium: "theatre", about: ["tragedy", "law", "justice", "family", "greek drama"] },
      { title: "The Oresteia", by: "Aeschylus", medium: "theatre", about: ["tragedy", "justice", "revenge", "greek drama"] },
      { title: "Medea", by: "Euripides", medium: "theatre", about: ["tragedy", "marriage", "revenge", "greek drama"] },
      { title: "The Bacchae", by: "Euripides", medium: "theatre", about: ["tragedy", "religion", "greek drama"] },
    ],
  },
  {
    id: "vatican-films-1995",
    name: "The Vatican's film list (1995)",
    about: "Films the Pontifical Council for Social Communications named for the centenary of cinema, under religion, values and art.",
    source: "Pontifical Council for Social Communications, Some Important Films, 1995",
    items: [
      { title: "The Passion of Joan of Arc (1928)", by: "Carl Theodor Dreyer", medium: "film", about: ["faith", "martyrdom", "religion"] },
      { title: "Ordet (1955)", by: "Carl Theodor Dreyer", medium: "film", about: ["faith", "miracles", "religion"] },
      { title: "Andrei Rublev (1966)", by: "Andrei Tarkovsky", medium: "film", about: ["art", "faith", "russia"] },
      { title: "A Man for All Seasons (1966)", by: "Fred Zinnemann", medium: "film", about: ["conscience", "law", "faith"] },
      { title: "The Seventh Seal (1957)", by: "Ingmar Bergman", medium: "film", about: ["death", "faith", "god"] },
      { title: "Bicycle Thieves (1948)", by: "Vittorio De Sica", medium: "film", about: ["poverty", "family"] },
      { title: "It's a Wonderful Life (1946)", by: "Frank Capra", medium: "film", about: ["virtue", "community", "family"] },
      { title: "Citizen Kane (1941)", by: "Orson Welles", medium: "film", about: ["power", "wealth", "art"] },
      { title: "2001: A Space Odyssey (1968)", by: "Stanley Kubrick", medium: "film", about: ["science", "human nature", "art"] },
      { title: "Schindler's List (1993)", by: "Steven Spielberg", medium: "film", about: ["holocaust", "virtue", "history"] },
    ],
  },
];

export const itemListById = (id: string) => ITEM_LISTS.find((l) => l.id === id);

/**
 * Items from the lists to start a way with, best matching the person's subjects first (by shared words), leaving
 * out titles they keep. Practice has no list yet.
 */
export function starters(subjects: readonly string[], have: ReadonlySet<string>): Record<Way, { title: string; by?: string; medium: Medium; list: string }[]> {
  const words = new Set(subjects.flatMap((s) => s.toLowerCase().split(/[^\p{L}]+/u)).filter((w) => w.length > 3));
  const out: Record<Way, { title: string; by?: string; medium: Medium; list: string; score: number }[]> = { teaching: [], dialectic: [], imitation: [], practice: [] };
  for (const l of ITEM_LISTS) for (const i of l.items) {
    if (have.has(i.title.toLowerCase())) continue;
    const score = i.about.reduce((n, a) => n + a.toLowerCase().split(/[^\p{L}]+/u).filter((w) => words.has(w)).length, 0);
    out[mediumOf(i.medium).way].push({ title: i.title, ...(i.by ? { by: i.by } : {}), medium: i.medium, list: l.name, score });
  }
  return Object.fromEntries(Object.entries(out).map(([w, xs]) => [w, xs.sort((a, b) => b.score - a.score).map(({ score: _s, ...x }) => x)])) as Record<Way, { title: string; by?: string; medium: Medium; list: string }[]>;
}
