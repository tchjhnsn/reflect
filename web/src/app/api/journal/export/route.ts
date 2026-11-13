import { currentUser } from "@/server/session";
import { exportJournal } from "@/server/journal-store";

export const dynamic = "force-dynamic";

/** The person's whole journal as one bundle file, to keep or move to another host. */
export async function GET() {
  const user = await currentUser();
  if (!user) return new Response("Sign in first.", { status: 401 });
  const bundle = await exportJournal(user.id);
  const day = new Date().toISOString().slice(0, 10);
  return new Response(JSON.stringify(bundle, null, 2), {
    headers: { "content-type": "application/json", "content-disposition": `attachment; filename="reflect-journal-${day}.json"` },
  });
}
