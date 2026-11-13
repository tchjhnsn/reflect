import type { HighlightColor } from "@reflect/core/excerpts";

/** How each highlight colour looks: the mark in the text, the bar beside an excerpt, and the dot to pick it. */
export const COLOR: Record<HighlightColor, { mark: string; bar: string; dot: string; name: string }> = {
  yellow: { mark: "bg-amber-200/70 dark:bg-amber-400/30", bar: "border-amber-300", dot: "bg-amber-300", name: "Yellow" },
  green: { mark: "bg-emerald-200/70 dark:bg-emerald-400/30", bar: "border-emerald-400", dot: "bg-emerald-400", name: "Green" },
  blue: { mark: "bg-sky-200/70 dark:bg-sky-400/30", bar: "border-sky-400", dot: "bg-sky-400", name: "Blue" },
  pink: { mark: "bg-pink-200/70 dark:bg-pink-400/30", bar: "border-pink-400", dot: "bg-pink-400", name: "Pink" },
};
