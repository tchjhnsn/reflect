"use client";

import { useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ArrowLeftIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

// The pages visited in this tab since the app loaded, so back can return to the last one instead of a fixed parent.
const trail: string[] = [];

/**
 * Back as a browser's back is: the page you came from, when you came from a page in Reflect; otherwise (a page opened
 * directly or after a reload) the page's parent.
 */
export function BackButton({ fallback, label }: { fallback: string; label: string }) {
  const router = useRouter();
  const path = usePathname();
  const query = useSearchParams().toString();
  const here = query ? `${path}?${query}` : path;
  useEffect(() => {
    // The browser's own back lands on the page before this one: step the trail back too, instead of adding it again.
    if (trail[trail.length - 2] === here) trail.pop();
    else if (trail[trail.length - 1] !== here) trail.push(here);
  }, [here]);
  return (
    <Button asChild variant="ghost" size="icon" aria-label={label}>
      <Link href={fallback} onClick={(e) => { if (trail.length > 1) { e.preventDefault(); trail.pop(); router.back(); } }}><ArrowLeftIcon /></Link>
    </Button>
  );
}
