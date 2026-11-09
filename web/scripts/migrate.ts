import { readFileSync } from "node:fs";
import { getMigrations } from "better-auth/db/migration";
import { auth } from "../src/server/auth";
import { db } from "../src/db/pool";

const { runMigrations } = await getMigrations(auth.options);
await runMigrations();
await db().query(readFileSync(new URL("../src/db/migrations/0001_person.sql", import.meta.url), "utf8"));
console.log("migrated");
await db().end();
