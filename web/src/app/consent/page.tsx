import type { Metadata } from "next";
import { db } from "@/db/pool";
import { Consent } from "./consent";

export const metadata: Metadata = { title: "Connect an app to Reflect" };

/** The name the app registered with. Apps name themselves, so the page says so. */
async function clientName(clientId: string): Promise<string | null> {
  if (!clientId) return null;
  const row = (await db().query(`SELECT name FROM "oauthClient" WHERE "clientId" = $1`, [clientId])).rows[0] as { name: string | null } | undefined;
  return row?.name ?? null;
}

export default async function ConsentPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const q = await searchParams;
  const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
  return <Consent name={await clientName(one(q.client_id))} scope={one(q.scope)} />;
}
