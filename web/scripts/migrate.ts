import { readdirSync, readFileSync } from "node:fs";
import { getMigrations } from "better-auth/db/migration";
import { auth } from "../src/server/auth";
import { db } from "../src/db/pool";

const { runMigrations } = await getMigrations(auth.options);
await runMigrations();
const dir = new URL("../src/db/migrations/", import.meta.url);
for (const file of readdirSync(dir).filter((f) => f.endsWith(".sql")).sort()) {
  await db().query(readFileSync(new URL(file, dir), "utf8"));
  console.log(`applied ${file}`);
}
await db().end();
