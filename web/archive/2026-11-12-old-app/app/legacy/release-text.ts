import type { ReleaseState } from "@/server/legacy-store";

const day = (iso: string) => new Date(iso).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });

/** A release's state in plain words. */
export function releaseText(r: ReleaseState): string {
  switch (r.state) {
    case "none":
      return "No release is under way.";
    case "confirming":
      return `${r.confirmed} of ${r.needed} trustees have confirmed a death. Nothing happens until ${r.needed} have.`;
    case "waiting":
      return `${r.confirmed} of ${r.needed} trustees have confirmed. The release happens on ${day(r.releasesAt)} unless it is stopped before then.`;
    case "released":
      return `The release happened on ${day(r.releasedAt)}.`;
  }
}
