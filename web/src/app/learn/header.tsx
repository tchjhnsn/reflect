import Link from "next/link";
import { SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SectionHeader } from "@/components/headers";
import { Segmented } from "@/components/segmented";

/**
 * The header of Learn's five sections: your own catalog (shelves, reading, highlights, claims) as tabs, and Browse
 * apart from them, since browsing is for what is not yours yet. One thing's page and a page where you are doing
 * something have their own headers (`@/components/headers`).
 */
export function LearnHeader({ on }: { on: "shelves" | "reading" | "claims" | "highlights" | "browse" }) {
  return (
    <SectionHeader back="/" backLabel="Home" title="Learn" tabs={
      <>
        <Segmented
          label="Your catalog"
          items={[
            { label: "My shelves", href: "/learn", on: on === "shelves" },
            { label: "Reading", href: "/learn/reading", on: on === "reading" },
            { label: "Highlights", href: "/learn/highlights", on: on === "highlights" },
            { label: "Claims", href: "/learn?tab=claims", on: on === "claims" },
          ]}
        />
        <Button asChild variant={on === "browse" ? "default" : "outline"} aria-current={on === "browse" ? "page" : undefined} className="shrink-0">
          <Link href="/learn/browse"><SearchIcon />Browse</Link>
        </Button>
      </>
    } />
  );
}
