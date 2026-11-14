/**
 * Watching and listening inside Reflect: which player a link plays in, and reading a transcript the person pasted
 * (YouTube's "Show transcript" copy, WebVTT or SRT) into timed parts that can be searched and quoted with their
 * moment. YouTube does not let other sites fetch a video's captions, so a transcript is pasted, never fetched.
 */
export type Embed = { kind: "youtube"; id: string; start: number } | { kind: "frame"; src: string; site: string; tall: boolean };
export type Part = { at: number; text: string };

/** "1:02:03", "12:34", "75" or "1m15s" in seconds, or null. */
export function seconds(s: string | null | undefined): number | null {
  if (!s) return null;
  const t = s.trim();
  const hms = /^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/.exec(t);
  if (hms && (hms[1] || hms[2] || hms[3])) return Number(hms[1] ?? 0) * 3600 + Number(hms[2] ?? 0) * 60 + Number(hms[3] ?? 0);
  const c = /^(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?$/.exec(t);
  if (c) return Number(c[1] ?? 0) * 3600 + Number(c[2]) * 60 + Number(c[3]);
  return /^\d+$/.test(t) ? Number(t) : null;
}

/** 75 as "1:15", 3723 as "1:02:03". */
export function clock(s: number): string {
  const n = Math.max(0, Math.floor(s));
  const h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), sec = n % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

/** The player a link plays in: YouTube (with its API, so the place is kept), else the site's own embed. */
export function embedOf(raw: string | null | undefined): Embed | null {
  if (!raw) return null;
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    return null;
  }
  const h = u.hostname.replace(/^(www|m)\./, "");
  if (h === "youtube.com" || h === "youtu.be" || h === "music.youtube.com") {
    const id = h === "youtu.be" ? u.pathname.slice(1) : u.searchParams.get("v") ?? /^\/(?:live|shorts|embed)\/([\w-]+)/.exec(u.pathname)?.[1] ?? "";
    return /^[\w-]{6,20}$/.test(id) ? { kind: "youtube", id, start: seconds(u.searchParams.get("t")) ?? 0 } : null;
  }
  const vimeo = h === "vimeo.com" ? /^\/(\d+)/.exec(u.pathname) : null;
  if (vimeo) return { kind: "frame", src: `https://player.vimeo.com/video/${vimeo[1]}`, site: "Vimeo", tall: true };
  const ia = h === "archive.org" ? /^\/(?:details|embed)\/([\w.-]+)/.exec(u.pathname) : null;
  if (ia) return { kind: "frame", src: `https://archive.org/embed/${ia[1]}`, site: "Internet Archive", tall: true };
  const sp = h === "open.spotify.com" ? /^\/(episode|show|track|album|playlist)\/(\w+)/.exec(u.pathname) : null;
  if (sp) return { kind: "frame", src: `https://open.spotify.com/embed/${sp[1]}/${sp[2]}`, site: "Spotify", tall: false };
  if (h === "soundcloud.com") return { kind: "frame", src: `https://w.soundcloud.com/player/?url=${encodeURIComponent(u.toString())}`, site: "SoundCloud", tall: false };
  return null;
}

/**
 * Reads a pasted transcript into timed parts. A line that is only a time ("12:34") starts a part (YouTube's copy);
 * "12:34 words" holds both; WebVTT and SRT cue times ("00:12:34.000 --> ...") start a part. Text with no times at
 * all is one part at 0.
 */
export function readTranscript(text: string): Part[] {
  const parts: Part[] = [];
  let cur: Part | null = null;
  const push = () => { if (cur && cur.text.trim()) parts.push({ at: cur.at, text: cur.text.replace(/\s+/g, " ").trim() }); };
  let fresh = true;
  for (const raw of text.replace(/\r/g, "").split("\n")) {
    const line = raw.trim();
    if (!line) { fresh = true; continue; }
    // A cue number in SRT or WebVTT stands alone after a blank line, before its times.
    if (line === "WEBVTT" || (fresh && /^\d+$/.test(line))) continue;
    fresh = false;
    const cue = /^(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->/.exec(line);
    const only = /^(\d{1,2}:\d{2}(?::\d{2})?)$/.exec(line);
    const lead = /^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$/.exec(line);
    if (cue || only) { push(); cur = { at: seconds((cue ?? only)![1]!.replace(/[.,]\d+$/, "")) ?? 0, text: "" }; continue; }
    if (lead) { push(); cur = { at: seconds(lead[1]!) ?? 0, text: lead[2]! }; continue; }
    if (!cur) cur = { at: 0, text: "" };
    cur.text += ` ${line}`;
  }
  push();
  return parts;
}

/** Parts whose words hold every word searched for, ignoring case, at most `limit`. */
export function searchTranscript(parts: readonly Part[], q: string, limit = 30): Part[] {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  return parts.filter((p) => { const t = p.text.toLowerCase(); return words.every((w) => t.includes(w)); }).slice(0, limit);
}
