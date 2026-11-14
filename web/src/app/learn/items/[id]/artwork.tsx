"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

/** IIIF images (the Art Institute of Chicago's) asked for at a given width; any other picture as it is. */
const sized = (src: string, w: number) => src.replace(/\/full\/[^/]+\/0\/default\.jpg$/, `/full/${w},/0/default.jpg`);

/** A work of art looked at inside Reflect: the picture large, and in full detail to pan across when zoomed. */
export function Artwork({ src, title }: { src: string; title: string }) {
  const [zoom, setZoom] = useState(false);
  return (
    <div className="space-y-2">
      <div className={`overflow-auto rounded-lg border bg-neutral-100 dark:bg-neutral-900 ${zoom ? "max-h-[80vh]" : ""}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={sized(src, zoom ? 3000 : 1200)} alt={title} className={zoom ? "max-w-none cursor-zoom-out" : "mx-auto max-h-[75vh] w-auto max-w-full cursor-zoom-in object-contain"} onClick={() => setZoom(!zoom)} />
      </div>
      <Button size="sm" variant="outline" onClick={() => setZoom(!zoom)}>{zoom ? "Fit to the page" : "See it in full detail"}</Button>
    </div>
  );
}
