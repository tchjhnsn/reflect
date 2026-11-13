// Learn's categories: Thema, the international subject scheme of the book trade (EDItEUR, v1.7, free to use
// under EDItEUR's licence; ns.editeur.org/thema). Its 20 sections are the parents and their two-letter
// categories the children. Headings are Thema's own; `short` is a shortened heading for a chip.
//
// Thema is assigned by publishers, and Open Library does not carry it, so Reflect places a book itself:
// first by the catalog's subjects for fiction and children's books (a call number files a novel by its
// author's country, not its genre), then by its Library of Congress call number through the crosswalk below,
// then by subjects again. The crosswalk is Reflect's own approximation, not EDItEUR's or the Library of
// Congress's. The call number stays on the book as its exact address. A book nothing places is not guessed.

export type LccRange = string; // "BF", "HM-HX" (letters), or "QA76-76", "TX641-840" (letters, whole numbers)
export type ThemaCategory = { code: string; heading: string; short: string; lcc?: LccRange[]; subjects?: string[] };
export type ThemaSection = { code: string; heading: string; short: string; children: ThemaCategory[] };

const c = (code: string, heading: string, short: string, lcc: LccRange[] = [], subjects: string[] = []): ThemaCategory => ({ code, heading, short, ...(lcc.length ? { lcc } : {}), ...(subjects.length ? { subjects } : {}) });

export const THEMA: ThemaSection[] = [
  { code: "A", heading: "The Arts", short: "Arts", children: [
    c("AB", "The arts: general topics", "Arts in general", ["NX"]),
    c("AF", "The Arts: art forms", "Art forms", ["N-NE"], ["painting", "sculpture"]),
    c("AJ", "Photography and photographs", "Photography", ["TR"], ["photography"]),
    c("AK", "Design, Industrial and commercial arts, illustration", "Design", ["NK"], ["design", "illustration"]),
    c("AM", "Architecture", "Architecture", ["NA"], ["architecture"]),
    c("AT", "Performing arts", "Performing arts", ["PN1560-9999", "GV1580-1799"], ["theater", "motion pictures", "dance"]),
    c("AV", "Music", "Music", ["M", "ML", "MT"], ["music"]),
  ] },
  { code: "C", heading: "Language and Linguistics", short: "Language", children: [
    c("CB", "Language: reference and general", "Language reference", ["PE"], ["english language"]),
    c("CF", "Linguistics", "Linguistics", ["P"], ["linguistics"]),
    c("CJ", "Language teaching and learning", "Learning languages", [], ["study and teaching"]),
  ] },
  { code: "D", heading: "Biography, Literature and Literary studies", short: "Biography and literature", children: [
    c("DB", "Ancient, classical and medieval texts", "Classical texts", ["PA"], ["classical literature"]),
    c("DC", "Poetry / Poems", "Poetry", [], ["poetry"]),
    c("DD", "Plays, playscripts, drama", "Plays", [], ["drama"]),
    c("DH", "Oral literature", "Oral literature", ["GR"], ["folklore"]),
    c("DN", "Biography and non-fiction prose or narratives", "Biography", ["CT"], ["biography", "autobiography"]),
    c("DS", "Literature: history and criticism", "Literary criticism", ["PN1-1559"], ["history and criticism"]),
  ] },
  { code: "F", heading: "Fiction", short: "Fiction", children: [
    c("FB", "Fiction: literary / general", "Literary fiction", [], ["literary fiction"]),
    c("FC", "Biographical fiction / autobiographical fiction", "Biographical fiction", [], ["biographical fiction"]),
    c("FD", "Speculative fiction", "Speculative fiction", [], ["speculative fiction", "dystopias"]),
    c("FF", "Crime and / or mystery fiction", "Crime and mystery", [], ["detective and mystery stories", "mystery fiction", "crime fiction"]),
    c("FG", "Sports fiction", "Sports fiction", [], ["sports stories"]),
    c("FH", "Thriller / suspense fiction", "Thriller", [], ["thrillers", "thriller", "suspense fiction"]),
    c("FJ", "Adventure / action or military fiction", "Adventure", [], ["adventure stories", "action and adventure", "war stories"]),
    c("FK", "Horror, ghost stories and supernatural fiction", "Horror", [], ["horror", "horror tales", "ghost stories"]),
    c("FL", "Science fiction", "Science fiction", [], ["science fiction"]),
    c("FM", "Fantasy", "Fantasy", [], ["fantasy", "fantasy fiction"]),
    c("FN", "Traditional or cultural fiction and true stories, tales and narratives", "Tales", [], ["fairy tales", "legends"]),
    c("FR", "Romance", "Romance", [], ["romance", "love stories"]),
    c("FS", "Family life fiction / stories about family", "Family life", [], ["domestic fiction", "family life"]),
    c("FT", "Family saga / generational saga fiction", "Family saga", [], ["family saga", "family sagas"]),
    c("FU", "Humorous fiction", "Humorous fiction", [], ["humorous fiction", "humorous stories"]),
    c("FV", "Historical fiction", "Historical fiction", [], ["historical fiction"]),
    c("FW", "Religious and spiritual fiction", "Religious fiction", [], ["christian fiction", "religious fiction"]),
  ] },
  { code: "G", heading: "Reference, Information and Interdisciplinary subjects", short: "Reference", children: [
    c("GB", "Encyclopaedias and reference works", "Reference works", ["AE", "AG"], ["encyclopedias and dictionaries"]),
    c("GL", "Library and information sciences / museology", "Library science", ["Z"], ["library science"]),
  ] },
  { code: "J", heading: "Society and Social Sciences", short: "Society", children: [
    c("JB", "Society and culture: general", "Society and culture", ["HN"], ["social conditions"]),
    c("JH", "Sociology and anthropology", "Sociology and anthropology", ["HM", "HQ-HT", "GN", "GT"], ["sociology", "anthropology"]),
    c("JK", "Social services and welfare, criminology", "Welfare and criminology", ["HV"], ["criminology"]),
    c("JM", "Psychology", "Psychology", ["BF"], ["psychology"]),
    c("JN", "Education / educational sciences / pedagogy", "Education", ["L-LT"], ["education"]),
    c("JP", "Politics and government", "Politics", ["J-JZ"], ["politics and government", "political science"]),
    c("JW", "Warfare and defence", "Warfare", ["U-UH", "V-VM"], ["military history"]),
  ] },
  { code: "K", heading: "Economics, Finance, Business and Management", short: "Business and economics", children: [
    c("KC", "Economics", "Economics", ["HB-HD", "HJ"], ["economics"]),
    c("KF", "Finance and accounting", "Finance", ["HG", "HF5601-5689"], ["finance", "personal finance", "accounting"]),
    c("KJ", "Business and management", "Business", ["HD28-70", "HF"], ["management", "business"]),
    c("KN", "Industry, industry sectors and industrial studies", "Industry", ["HD9000-9999", "HE"]),
  ] },
  { code: "L", heading: "Law", short: "Law", children: [
    c("LA", "Jurisprudence and general issues", "Jurisprudence", ["K"], ["jurisprudence"]),
    c("LB", "International law", "International law", ["KZ"], ["international law"]),
    c("LN", "Laws of specific jurisdictions and specific areas of law", "Laws of a jurisdiction", ["KB-KY"], ["law"]),
  ] },
  { code: "M", heading: "Medicine and Nursing", short: "Medicine", children: [
    c("MB", "Medicine / Healthcare: general issues / topics", "Medicine in general", ["R", "RA"], ["medicine"]),
    c("MF", "Pre-clinical medicine: basic sciences", "Basic medical sciences", ["QM", "QP"], ["anatomy", "physiology"]),
    c("MJ", "Clinical and internal medicine", "Clinical medicine", ["RC"]),
    c("MK", "Medical specialties, branches of medicine", "Medical specialties", ["RE-RL", "RM"]),
    c("MN", "Surgery", "Surgery", ["RD"], ["surgery"]),
    c("MQ", "Nursing and ancillary services", "Nursing", ["RT"], ["nursing"]),
    c("MX", "Complementary, integrated and alternative medicine and therapies", "Alternative medicine", ["RZ"], ["alternative medicine"]),
    c("MZ", "Veterinary medicine", "Veterinary medicine", ["SF600-1100"], ["veterinary medicine"]),
  ] },
  { code: "N", heading: "History and Archaeology", short: "History", children: [
    c("NH", "History", "History", ["C-FZ"], ["history"]),
    c("NK", "Archaeology", "Archaeology", ["CC"], ["archaeology"]),
  ] },
  { code: "P", heading: "Mathematics and Science", short: "Science and math", children: [
    c("PB", "Mathematics", "Mathematics", ["QA"], ["mathematics"]),
    c("PD", "Science: general issues", "Science in general", ["Q"], ["science"]),
    c("PG", "Astronomy, space and time", "Astronomy", ["QB"], ["astronomy"]),
    c("PH", "Physics", "Physics", ["QC"], ["physics"]),
    c("PN", "Chemistry", "Chemistry", ["QD"], ["chemistry"]),
    c("PS", "Biology, life sciences", "Biology", ["QH-QR"], ["biology", "evolution"]),
  ] },
  { code: "Q", heading: "Philosophy and Religion", short: "Philosophy and religion", children: [
    c("QD", "Philosophy", "Philosophy", ["B-BD", "BH-BJ"], ["philosophy", "ethics"]),
    c("QR", "Religion and beliefs", "Religion", ["BL-BX"], ["religion", "christianity", "theology", "bible"]),
  ] },
  { code: "R", heading: "Earth Sciences, Geography, Environment, Planning", short: "Earth and environment", children: [
    c("RB", "Earth sciences", "Earth sciences", ["QE"], ["geology"]),
    c("RG", "Geography", "Geography", ["G-GB"], ["geography"]),
    c("RN", "The environment", "Environment", ["GE"], ["environment", "climate change"]),
  ] },
  { code: "S", heading: "Sports and Active outdoor recreation", short: "Sports", children: [
    c("SC", "Sport: general", "Sport in general", ["GV557-860"], ["sports"]),
    c("SF", "Ball sports / ball games", "Ball sports", ["GV861-1017"], ["baseball", "football", "basketball", "soccer"]),
    c("SR", "Combat sports and self-defence", "Combat sports", ["GV1100-1150"], ["martial arts", "boxing"]),
    c("SZ", "Active outdoor pursuits and activities", "Outdoors", ["GV191-200"], ["outdoor recreation", "hiking"]),
  ] },
  { code: "T", heading: "Technology, Engineering, Agriculture, Industrial processes", short: "Technology", children: [
    c("TB", "Technology: general issues", "Technology in general", ["T"], ["technology"]),
    c("TD", "Industrial chemistry and manufacturing technologies", "Manufacturing", ["TP", "TS"]),
    c("TG", "Mechanical engineering and materials", "Mechanical engineering", ["TJ"], ["mechanical engineering"]),
    c("TJ", "Electronics and communications engineering", "Electronics", ["TK"], ["electronics", "electrical engineering"]),
    c("TN", "Civil engineering, surveying and building", "Civil engineering", ["TA", "TC", "TE", "TG", "TH"], ["civil engineering", "building"]),
    c("TQ", "Environmental science, engineering and technology", "Environmental engineering", ["TD"]),
    c("TR", "Transport technology and trades", "Transport technology", ["TF", "TL"], ["automobiles", "aeronautics"]),
    c("TV", "Agriculture and farming", "Agriculture", ["S-SK"], ["agriculture"]),
  ] },
  { code: "U", heading: "Computing and Information Technology", short: "Computing", children: [
    c("UB", "Digital and Information technology: general topics", "Information technology", [], ["information technology"]),
    c("UM", "Computer programming / software engineering", "Programming", [], ["computer programming", "software engineering"]),
    c("UR", "Computer security", "Computer security", [], ["computer security"]),
    c("UY", "Computer science", "Computer science", ["QA75-76"], ["computer science", "artificial intelligence"]),
  ] },
  { code: "V", heading: "Health, Relationships and Personal development", short: "Health and personal growth", children: [
    c("VF", "Family and health", "Family and health", ["RA773-790", "HQ755-759"], ["parenting", "health"]),
    c("VS", "Self-help, personal development and practical advice", "Self-help", ["BF632-637"], ["self-help", "self-actualization (psychology)", "success"]),
    c("VX", "Mind, body, spirit", "Mind, body, spirit", ["BF1001-1999"], ["spirituality", "meditation"]),
  ] },
  { code: "W", heading: "Lifestyle, Hobbies and Leisure", short: "Lifestyle and hobbies", children: [
    c("WB", "Cookery / food and drink / food writing / cookbooks", "Food and cooking", ["TX641-840"], ["cooking", "cookbooks"]),
    c("WD", "Hobbies, quizzes and games", "Hobbies and games", ["GV1199-1570"], ["games"]),
    c("WF", "Handicrafts, decorative arts and crafts", "Crafts", ["TT"], ["handicraft"]),
    c("WH", "Humour", "Humour", ["PN6147-6231"], ["humor", "wit and humor"]),
    c("WM", "Gardening", "Gardening", ["SB450-466"], ["gardening"]),
    c("WN", "Nature and the natural world: general interest / nature writing", "Nature", [], ["nature", "natural history"]),
    c("WT", "Travel and holiday", "Travel", ["G149-570"], ["travel", "description and travel"]),
  ] },
  { code: "X", heading: "Graphic novels, Comic books, Manga, Cartoons", short: "Graphic novels and comics", children: [
    c("XQ", "Comics (graphic novel / comic book / manga etc): genres", "Comics and manga", ["PN6700-6790"], ["graphic novels", "comic books, strips, etc", "manga"]),
  ] },
  { code: "Y", heading: "Children’s, Teenage and Educational", short: "Children and teens", children: [
    c("YB", "Children’s: picture books, activity books, early learning concepts", "Picture books", [], ["picture books"]),
    c("YF", "Children’s / Teenage fiction and true stories", "Children's fiction", ["PZ"], ["juvenile fiction", "young adult fiction"]),
    c("YN", "Children’s / Teenage: General interest", "Children's nonfiction", [], ["juvenile literature"]),
  ] },
];

const SECTIONS = new Map(THEMA.map((s) => [s.code, s]));
const CATEGORIES = new Map(THEMA.flatMap((s) => s.children.map((x) => [x.code, x] as const)));
export const themaSection = (code: string): ThemaSection | undefined => SECTIONS.get(code.slice(0, 1).toUpperCase());
export const themaCategory = (code: string): ThemaCategory | undefined => CATEGORIES.get(code.toUpperCase());

// --- Library of Congress call numbers -------------------------------------------------------------

type Call = { letters: string; number: number };
/** "BF637.S8 C55", "BF-0637.00000000.S8" and "B--0383" all read as letters and the whole class number. */
export function readCall(callNumber: string): Call | null {
  const m = /^([A-Z]{1,3})-{0,2}0*(\d+)?/.exec(callNumber.trim().toUpperCase());
  if (!m || /[IOWXY]/.test(m[1]![0]!)) return null;
  return { letters: m[1]!, number: m[2] ? Number(m[2]) : 0 };
}

/** A call number as a library prints it: Open Library's "BF-0335.30000000.C54" reads "BF335.3 .C54". */
export function showCall(callNumber: string): string {
  if (!callNumber.includes("-")) return callNumber.trim();
  const m = /^([A-Z]{1,3})-*0*(\d+)(?:\.(\d+))?(?:\.(.+))?$/.exec(callNumber.trim().toUpperCase());
  if (!m) return callNumber.trim();
  const dec = (m[3] ?? "").replace(/0+$/, "");
  return `${m[1]}${m[2]}${dec ? `.${dec}` : ""}${m[4] ? ` .${m[4]}` : ""}`;
}

type Range = { from: string; to: string; lo: number; hi: number };
function readRange(r: LccRange): Range {
  const n = /^([A-Z]{1,3})(\d+)-(\d+)$/.exec(r);
  if (n) return { from: n[1]!, to: n[1]!, lo: Number(n[2]), hi: Number(n[3]) };
  const [from, to = from] = r.split("-") as [string, string?];
  return { from, to: to!, lo: 0, hi: 99999 };
}
const inLetters = (l: string, r: Range) => (r.from === r.to ? l === r.from : l >= r.from && l <= r.to);
/** Narrower ranges win: a number span before one class, one class before a span of classes. */
const width = (r: Range) => (r.hi - r.lo < 99999 ? r.hi - r.lo : 1e6 + (r.from === r.to ? 0 : 1e6));
const BY_CALL = THEMA.flatMap((s) => s.children.flatMap((x) => (x.lcc ?? []).map((r) => ({ code: x.code, range: readRange(r) }))))
  .sort((a, b) => width(a.range) - width(b.range));

/** The Thema category one call number points to through the crosswalk, or null. */
export function categoryOfCall(callNumber: string): string | null {
  const call = readCall(callNumber);
  if (!call) return null;
  return BY_CALL.find(({ range }) => inLetters(call.letters, range) && call.number >= range.lo && call.number <= range.hi)?.code ?? null;
}

// --- Subjects ----------------------------------------------------------------------------------------

const said = (subjects: readonly string[], term: string) => subjects.some((s) => new RegExp(`(^|[^a-z])${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^a-z]|$)`).test(s.toLowerCase()));
/** The category of the first subject that names one, in the catalog's order (its first subjects lead). */
function bySubject(subjects: readonly string[], sections: string): string | null {
  const cats = THEMA.filter((s) => sections.includes(s.code)).flatMap((s) => s.children);
  for (const one of subjects) {
    const hit = cats.find((x) => (x.subjects ?? []).some((t) => said([one], t)));
    if (hit) return hit.code;
  }
  return null;
}

/**
 * Where a book belongs: children's books and fiction by their subjects, then the call number most of its
 * call numbers point to, then any category whose subjects it carries, then plain "fiction". Null when none.
 */
export function categoryOf(book: { lcc?: readonly string[]; subjects?: readonly string[] }): string | null {
  const subjects = book.subjects ?? [];
  // Catalog subjects tag many adult books "juvenile"; a call number outweighs them when there is one.
  const calls = (book.lcc ?? []).map(readCall).filter((x): x is Call => Boolean(x));
  const young = calls.length
    ? calls.filter((x) => x.letters === "PZ").length * 2 > calls.length
    : ["juvenile fiction", "juvenile literature", "young adult fiction", "picture books"].some((t) => said(subjects, t));
  if (young) return bySubject(subjects, "Y") ?? "YF";
  const fiction = said(subjects, "fiction") || said(subjects, "novel") || said(subjects, "stories");
  if (fiction) {
    const genre = bySubject(subjects, "F");
    if (genre) return genre;
  }
  const votes = new Map<string, number>();
  let best: string | null = null;
  for (const n of book.lcc ?? []) {
    const code = categoryOfCall(n);
    if (!code) continue;
    votes.set(code, (votes.get(code) ?? 0) + 1);
    if (!best || votes.get(code)! > votes.get(best)!) best = code;
  }
  if (best && !(fiction && best.startsWith("D"))) return best;
  if (fiction) return "FB";
  return best ?? bySubject(subjects, THEMA.map((s) => s.code).filter((x) => x !== "F" && x !== "Y").join(""));
}

// --- Open Library queries ----------------------------------------------------------------------------

/** Open Library stores call numbers as "BF-0637.00000000"; a range query over that form. */
function lccQuery(r: Range): string {
  const pad = (l: string) => (l.length === 1 ? `${l}-` : l);
  const num = (n: number) => String(Math.min(n, 9999)).padStart(4, "0");
  return `lcc:[${pad(r.from)}-${num(r.lo)} TO ${pad(r.to)}-${num(r.hi)}.99999999]`;
}

/** The Open Library search for a category (two letters) or a whole section (one letter). */
export function themaQuery(code: string): string | null {
  const cats = code.length === 1 ? themaSection(code)?.children ?? [] : [themaCategory(code)].filter((x): x is ThemaCategory => Boolean(x));
  // Call numbers are precise and subjects are loose tags, so a category with call numbers is searched by them.
  const parts = cats.flatMap((x) => (x.lcc?.length ? x.lcc.map((r) => lccQuery(readRange(r))) : (x.subjects ?? []).map((t) => `subject:"${t}"`)));
  if (!parts.length) return null;
  const any = parts.join(" OR ");
  return code.startsWith("F") ? `(${any}) -subject:"juvenile fiction" -lcc:[PZ-0000 TO PZ-9999.99999999]` : any;
}

// --- Shelves -----------------------------------------------------------------------------------------

export type ThemaBook = { id: string; title: string; lcc?: readonly string[]; catalogSubjects?: readonly string[]; subjects: readonly string[] };
export type ThemaShelf<B, S> = { id: string; name: string; books: (B & { category: string })[]; subjects: S[] };
export type ThemaShelves<B, S> = { shelves: ThemaShelf<B, S>[]; unshelved: { books: B[]; subjects: S[] } };

/**
 * Books on the section their category is in; a subject on the section most of its books are on. `all` keeps
 * the 20 sections, empty ones included, in Thema's order; otherwise only filled ones are kept.
 */
export function shelveThema<B extends ThemaBook, S extends { id: string; name: string }>(books: readonly B[], subjects: readonly S[], all = false): ThemaShelves<B, S> {
  const by = new Map(THEMA.map((s) => [s.code, { books: [] as (B & { category: string })[], subjects: [] as S[] }]));
  const unshelved = { books: [] as B[], subjects: [] as S[] };
  const votes = new Map<string, Map<string, number>>();
  for (const b of books) {
    const category = categoryOf({ ...(b.lcc ? { lcc: b.lcc } : {}), subjects: b.catalogSubjects ?? [] });
    if (!category) {
      unshelved.books.push(b);
      continue;
    }
    const section = category[0]!;
    by.get(section)!.books.push({ ...b, category });
    for (const s of b.subjects) {
      const v = votes.get(s) ?? new Map<string, number>();
      v.set(section, (v.get(section) ?? 0) + 1);
      votes.set(s, v);
    }
  }
  for (const s of subjects) {
    const v = votes.get(s.id);
    const best = v ? THEMA.map((x) => x.code).reduce<string | null>((a, code) => ((v.get(code) ?? 0) > (a ? v.get(a) ?? 0 : 0) ? code : a), null) : null;
    (best ? by.get(best)!.subjects : unshelved.subjects).push(s);
  }
  const shelves = THEMA.map((s) => ({ id: s.code, name: s.short, ...by.get(s.code)! })).filter((s) => all || s.books.length || s.subjects.length);
  return { shelves, unshelved };
}
