"use client";

// YouTube's player API, loaded once, and the person's own date, for players that keep a place.
export type YTPlayer = { getCurrentTime(): number; seekTo(s: number, allow: boolean): void; playVideo(): void; destroy(): void };
type YTNamespace = { Player: new (el: HTMLElement, o: Record<string, unknown>) => YTPlayer; PlayerState: { PAUSED: number; ENDED: number } };
declare global {
  interface Window { YT?: YTNamespace; onYouTubeIframeAPIReady?: () => void }
}

export const localDay = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };

export function loadYouTube(): Promise<YTNamespace> {
  return new Promise((resolve) => {
    if (window.YT?.Player) return resolve(window.YT);
    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => { prev?.(); resolve(window.YT!); };
    if (!document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) {
      const s = document.createElement("script");
      s.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(s);
    }
  });
}

