export const THEMES = [
  { id: "paper", title: "Paper", about: "Warm paper and serif type." },
  { id: "stream", title: "Stream", about: "Dark, like a microblog." },
  { id: "calm", title: "Calm", about: "Soft colour and rounded cards." },
] as const;
export type ThemeId = (typeof THEMES)[number]["id"];
export const DEFAULT_THEME: ThemeId = "paper";
export const THEME_COOKIE = "reflect-theme";
export const asTheme = (v: string | undefined): ThemeId => (THEMES.some((t) => t.id === v) ? (v as ThemeId) : DEFAULT_THEME);
