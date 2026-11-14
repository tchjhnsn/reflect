// Where a thing sits, for its header's breadcrumbs: on your shelves under its way when you keep it, else in Browse
// under its medium. The rule is fixed, so the same thing always says the same place.
import { BROWSE_MEDIA, isBrowseMedium } from "@reflect/core/browse-media";
import { LEISURE, WAYS, type Way } from "@reflect/core/ways";
import type { Crumb } from "@/components/headers";

export const LEARN: Crumb = { label: "Learn", href: "/learn" };

export function browseCrumbs(medium?: string | null): Crumb[] {
  const m = isBrowseMedium(medium) ? BROWSE_MEDIA.find((x) => x.id === medium)! : null;
  return [LEARN, { label: "Browse", href: "/learn/browse" }, ...(m ? [{ label: m.label, href: `/learn/browse?m=${m.id}` }] : [])];
}

export function shelfCrumbs(place?: Way | "leisure" | null): Crumb[] {
  const p = place === "leisure" ? LEISURE : WAYS.find((w) => w.id === place);
  return [LEARN, { label: "My shelves", href: "/learn" }, ...(p ? [{ label: p.label, href: `/learn#${p.id}` }] : [])];
}
