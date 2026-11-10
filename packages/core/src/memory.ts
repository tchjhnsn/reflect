// Memory (0 in the table): the append-only log of every write to an instance, hash-chained so that
// a removed or changed line is found. In a repository it is memory.jsonl at the instance root.
import { createHash } from "node:crypto";

export type MemoryLine = {
  seq: number;
  at: string;
  op: "create" | "append";
  /** The record's path, relative to the instance root. */
  path: string;
  /** sha256 of the record's bytes after the write. */
  sha256: string;
  /** hash of the line before, or null for the first. */
  prev: string | null;
  hash: string;
};

export const sha256 = (s: string): string => createHash("sha256").update(s).digest("hex");

const hashOf = (l: Omit<MemoryLine, "hash">): string => sha256(JSON.stringify([l.seq, l.at, l.op, l.path, l.sha256, l.prev]));

/** The next line after `last`. */
export function nextLine(last: MemoryLine | null, op: MemoryLine["op"], path: string, bytes: string, at: string): MemoryLine {
  const line = { seq: last ? last.seq + 1 : 0, at, op, path, sha256: sha256(bytes), prev: last ? last.hash : null };
  return { ...line, hash: hashOf(line) };
}

export function parseMemory(text: string): MemoryLine[] {
  return text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as MemoryLine);
}

/** Problems with the chain, empty when it holds. */
export function verifyMemory(lines: readonly MemoryLine[]): string[] {
  const out: string[] = [];
  lines.forEach((l, i) => {
    const prev = i === 0 ? null : lines[i - 1]!.hash;
    if (l.seq !== i) out.push(`line ${i}: seq is ${l.seq}`);
    if (l.prev !== prev) out.push(`line ${i}: does not follow the line before`);
    const { hash, ...rest } = l;
    if (hashOf(rest) !== hash) out.push(`line ${i}: hash does not match its fields`);
  });
  return out;
}

/** The last recorded sha256 of each path. */
export function lastShas(lines: readonly MemoryLine[]): Map<string, string> {
  const m = new Map<string, string>();
  for (const l of lines) m.set(l.path, l.sha256);
  return m;
}
