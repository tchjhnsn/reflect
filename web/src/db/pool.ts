import { Pool, neonConfig } from "@neondatabase/serverless";
import ws from "ws";

neonConfig.webSocketConstructor = ws;

let pool: Pool | null = null;

export function db(): Pool {
  if (!pool) {
    const url = process.env.DATABASE_URL;
    if (!url && process.env.NEXT_PHASE !== "phase-production-build") throw new Error("DATABASE_URL is not set");
    pool = new Pool({ connectionString: url });
  }
  return pool;
}
