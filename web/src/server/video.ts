// Video calls inside Reflect, through Daily (daily.co): one private room per examination, joined only with a token
// Reflect makes for each person, so the link alone lets nobody in. Nothing is recorded. Needs DAILY_API_KEY; without
// it the call is not shown.
const API = "https://api.daily.co/v1";

export const videoReady = (): boolean => Boolean(process.env.DAILY_API_KEY);

async function daily<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T | null> {
  const key = process.env.DAILY_API_KEY;
  if (!key) return null;
  try {
    const r = await fetch(`${API}${path}`, {
      method: init.method ?? "GET", cache: "no-store", signal: AbortSignal.timeout(8000),
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      ...(init.body ? { body: JSON.stringify(init.body) } : {}),
    });
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

const roomName = (id: string) => `reflect-exam-${id.replace(/-/g, "").slice(0, 24)}`;

/**
 * A place in the examination's call for one person: the room (made the first time anyone joins) and a token naming
 * them, good for six hours. The candidate holds the room (can remove someone). Null when video is not set up.
 */
export async function callFor(examinationId: string, person: { name: string; owner: boolean }): Promise<{ url: string } | null> {
  const name = roomName(examinationId);
  const room = (await daily<{ url: string }>(`/rooms/${name}`)) ?? (await daily<{ url: string }>("/rooms", { method: "POST", body: { name, privacy: "private", properties: { enable_prejoin_ui: true, max_participants: 14 } } }));
  if (!room?.url) return null;
  const exp = Math.floor(Date.now() / 1000) + 6 * 3600;
  const token = await daily<{ token: string }>("/meeting-tokens", { method: "POST", body: { properties: { room_name: name, user_name: person.name, is_owner: person.owner, exp } } });
  return token ? { url: `${room.url}?t=${encodeURIComponent(token.token)}` } : null;
}

/** Ends the examination's call for good once the examination is closed. */
export async function endCall(examinationId: string): Promise<void> {
  await daily(`/rooms/${roomName(examinationId)}`, { method: "DELETE" });
}
