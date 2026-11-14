// Podcasts heard in Reflect: a show's episodes read from its own feed (@reflect/core/podcasts), cached an hour by
// Next's fetch cache where the feed is small enough, and never stored in the database.
import { episodeTranscript, showOf, type Episode, type Show } from "@reflect/core/podcasts";
import type { Passage } from "@reflect/core/courses";

const page = async (url: string): Promise<string | null> => {
  try {
    const r = await fetch(url, { headers: { "user-agent": "Reflect podcast player" }, next: { revalidate: 3600 }, signal: AbortSignal.timeout(10000) });
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
};

export const showFor = (url: string | null): Promise<Show | null> => showOf(url, page);
export const transcriptOf = (e: Episode): Promise<Passage[]> => episodeTranscript(e, page);
