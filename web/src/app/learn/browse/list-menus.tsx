import { SORTS, type Sort } from "@reflect/core/openlibrary";
import { FILTERS, type Filter } from "@/server/browse";
import { Menu } from "./category-menu";

/** The settings a list's address carries beside what it lists. */
export type ListState = { sort?: Sort; filter?: Filter; hide: boolean };
const url = (path: string, base: Record<string, string>, s: ListState) =>
  `${path}?${new URLSearchParams({ ...base, ...(s.sort ? { sort: s.sort } : {}), ...(s.filter ? { filter: s.filter } : {}), ...(s.hide ? { hide: "1" } : {}) })}`.replace(/\?$/, "");

/** The orders a list can be in, as a menu; a search can also be in best-match order, its default. */
export function SortMenu({ path, base, state, match = false, fallback = "rating" }: { path: string; base: Record<string, string>; state: ListState; match?: boolean; fallback?: Sort }) {
  const items = (Object.keys(SORTS) as Sort[]).filter((s) => match || s !== fallback).map((s) => ({ href: url(path, base, { ...state, sort: s }), label: SORTS[s], on: s === state.sort }));
  const first = { href: url(path, base, { ...state, sort: undefined }), label: match ? "Best match" : SORTS[fallback], on: !state.sort };
  return <Menu label="Sort" items={[first, ...items]} />;
}

/** Narrowing a list, and leaving out the books already on the person's shelves, as one menu. */
export function FilterMenu({ path, base, state }: { path: string; base: Record<string, string>; state: ListState }) {
  const items = [
    { href: url(path, base, { ...state, filter: undefined }), label: "All books", on: !state.filter },
    ...(Object.keys(FILTERS) as Filter[]).map((f) => ({ href: url(path, base, { ...state, filter: f }), label: FILTERS[f], on: f === state.filter })),
    { href: url(path, base, { ...state, hide: !state.hide }), label: "Hide books on my shelves", on: state.hide },
  ];
  const shown = [state.filter ? FILTERS[state.filter] : "All books", state.hide ? "not on my shelves" : null].filter(Boolean).join(", ");
  return <Menu label="Filter" items={items} lead={Object.keys(FILTERS).length + 1} shown={shown} />;
}
