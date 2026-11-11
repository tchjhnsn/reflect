// Write one record into an instance folder and log it in memory.jsonl. The folder host's journal has
// its own copy of this for entries; everything else written through reflect comes here.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { nextLine, parseMemory, type MemoryLine } from "./memory.ts";

export async function readRecord(root: string, rel: string): Promise<string | null> {
  try {
    return await readFile(join(root, rel), "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw e;
  }
}

export async function writeRecord(root: string, rel: string, bytes: string, op: MemoryLine["op"], at: string): Promise<void> {
  await mkdir(dirname(join(root, rel)), { recursive: true });
  const lines = parseMemory((await readRecord(root, "memory.jsonl")) ?? "");
  await writeFile(join(root, rel), bytes);
  const line = nextLine(lines.at(-1) ?? null, op, rel, bytes, at);
  await writeFile(join(root, "memory.jsonl"), `${[...lines, line].map((l) => JSON.stringify(l)).join("\n")}\n`);
}
