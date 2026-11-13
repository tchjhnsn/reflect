import Link from "next/link";
import { SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/page-header";
import { Segmented } from "@/components/segmented";

/**
 * Learn's one header on every Learn page: your own catalog (shelves, highlights, claims) as tabs, and Browse apart from
 * them, since browsing is for books that are not yours yet.
 */
export function LearnHeader({ on }: { on: "shelves" | "claims" | "highlights" | "browse" }) {
  return (
    <PageHeader back="/" backLabel="Home" title="Learn">
      <Segmented
        label="Your catalog"
        items={[
          { label: "My shelves", href: "/learn", on: on === "shelves" },
          { label: "Highlights", href: "/learn/highlights", on: on === "highlights" },
          { label: "Claims", href: "/learn?tab=claims", on: on === "claims" },
        ]}
      />
      <Button asChild variant={on === "browse" ? "default" : "outline"} aria-current={on === "browse" ? "page" : undefined}>
        <Link href="/learn/browse"><SearchIcon />Browse books</Link>
      </Button>
    </PageHeader>
  );
}
