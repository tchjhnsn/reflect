// A word looked up while reading: its English senses from Wiktionary's wikitext (CC BY-SA), each with its usage
// labels, so older meanings ("archaic", "obsolete", "dated") can be shown as such. Pure parsing; the fetch is the
// caller's.

const POS = new Set(["Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Preposition", "Conjunction", "Interjection", "Determiner", "Article", "Numeral", "Particle", "Proper noun", "Contraction", "Phrase", "Prepositional phrase", "Proverb", "Idiom"]);
const OLD = /^(archaic|obsolete|dated|historical|poetic|literary)$/;

/** One sense: what it means, its labels ("archaic", "transitive"), whether it is an older meaning, and the word it is a form of. */
export type Sense = { text: string; labels: string[]; old: boolean; formOf?: string };
export type Definition = { pos: string; senses: Sense[] };

const linkText = (s: string) => s.replace(/\[\[(?:[^\]|]*\|)?([^\]]*)\]\]/g, "$1");

function plain(s: string): string {
  let t = s.replace(/<ref[^>]*\/>/g, "").replace(/<ref[\s\S]*?<\/ref>/g, "").replace(/<[^>]+>/g, "");
  // Templates that carry words: a gloss, a link, a Wikipedia link, a non-gloss definition.
  for (let i = 0; i < 3; i++) {
    t = t
      .replace(/\{\{(?:gloss|gl)\|([^{}|]*)\}\}/g, "($1)")
      .replace(/\{\{(?:l|m|w|ll|non-gloss definition|n-g|ngd|taxlink|vern)\|(?:en\|)?([^{}|]*)(?:\|[^{}]*)?\}\}/g, "$1")
      .replace(/\{\{[^{}]*\}\}/g, "");
  }
  return linkText(t).replace(/'''?/g, "").replace(/\s+/g, " ").replace(/\s+([,.;:])/g, "$1").replace(/^[\s,;:.]+/, "").trim();
}

/**
 * The English senses in a Wiktionary page's wikitext, by part of speech in page order. A sense that only says the
 * word is a form of another ("past participle of acknowledge") carries that word in `formOf`.
 */
export function definitionsOf(wikitext: string): Definition[] {
  const start = wikitext.search(/^==English==\s*$/m);
  if (start < 0) return [];
  const rest = wikitext.slice(start + 11);
  const end = rest.search(/^==[^=].*==\s*$/m);
  const english = end < 0 ? rest : rest.slice(0, end);
  const out: Definition[] = [];
  let current: Definition | null = null;
  for (const line of english.split("\n")) {
    const head = /^={3,5}\s*([^=]+?)\s*={3,5}\s*$/.exec(line);
    if (head) {
      current = POS.has(head[1]!) ? { pos: head[1]!, senses: [] } : null;
      if (current) out.push(current);
      continue;
    }
    if (!current || !/^#(?![:*#])/.test(line)) continue;
    const body = line.replace(/^#\s*/, "");
    const labels: string[] = [];
    for (const m of body.matchAll(/\{\{(?:lb|lbl|label)\|en\|([^{}]*)\}\}/g)) {
      for (const l of m[1]!.split("|")) if (l && !/^(_|or|and|en|also)$/.test(l) && !l.includes("=")) labels.push(l.replace(/_/g, " "));
    }
    const form = /\{\{([a-z][a-z -]*? of)\|en\|([^{}|]+)[^{}]*\}\}/.exec(body);
    const text = form ? `${form[1]![0]!.toUpperCase()}${form[1]!.slice(1)} ${linkText(form[2]!)}` : plain(body.replace(/\{\{(?:lb|lbl|label)\|en\|[^{}]*\}\}/g, ""));
    if (!text) continue;
    const sense: Sense = { text, labels, old: labels.some((l) => OLD.test(l)) };
    if (form) sense.formOf = linkText(form[2]!);
    current.senses.push(sense);
  }
  return out.filter((d) => d.senses.length);
}

/** The word to look up from a selection: one word of letters (an apostrophe or hyphen inside), or null. */
export function lookupWord(selection: string): string | null {
  const w = selection.trim().replace(/^[^\p{L}]+|[^\p{L}]+$/gu, "").replace(/[’]/g, "'");
  return /^\p{L}+(?:['-]\p{L}+)*$/u.test(w) && w.length <= 40 ? w : null;
}
