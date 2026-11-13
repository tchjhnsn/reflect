"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { HeadphonesIcon } from "lucide-react";
import type { Audiobook } from "@reflect/core/librivox";
import { Button } from "@/components/ui/button";
import { useListen } from "../../../player";

/**
 * Read and listen together: plays the recorded part that reads this chapter and has the text follow it, scrolled
 * by how far through the part the recording is, turning to the next chapter when the next part starts. Shown only
 * where the recording's parts line up with the text's chapters.
 */
export function ListenAlong({ book, work, title, map, chapter }: { book: Audiobook; work: string; title: string; /** For each chapter, the recorded part that reads it. */ map: (number | null)[]; chapter: number }) {
  const listen = useListen();
  const router = useRouter();
  const last = useRef(-1);
  const section = map[chapter - 1] ?? null;
  const following = Boolean(listen?.follow && listen.book?.id === book.id);

  useEffect(() => {
    if (!listen || !following) return;
    const text = map.indexOf(listen.chapter) + 1;
    if (text > 0 && text !== chapter) {
      router.push(`/learn/works/${work}/read?c=${text}`);
      return;
    }
    if (text !== chapter || !listen.duration) return;
    const y = (listen.time / listen.duration) * (document.documentElement.scrollHeight - window.innerHeight);
    if (Math.abs(y - last.current) > 40) {
      last.current = y;
      window.scrollTo({ top: y, behavior: "smooth" });
    }
  }, [listen, following, map, chapter, work, router]);

  if (!listen || section === null) return null;
  if (following) return <Button variant="outline" size="sm" onClick={() => listen.setFollow(false)}><HeadphonesIcon />Stop following</Button>;
  return (
    <Button variant="outline" size="sm" onClick={() => listen.play({ ...book, work, bookTitle: title }, section, { follow: true })}>
      <HeadphonesIcon />Listen along
    </Button>
  );
}
