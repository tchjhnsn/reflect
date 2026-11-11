// Sending an idea to Naive, the venture development framework, on the person's explicit say-so. Reflect is
// where an idea starts; Naive is where it becomes a venture. Reflect never imports Naive: it runs the
// person's own `naive capture` in their Naive workspace (a folder with naive-workspace.json), which keeps the
// text in the workspace's raw/ and files it in the venture it fits or the workspace inbox. The idea note then
// gets a dated `sent to naive` line naming where the text landed, so the two stay linked both ways: the raw
// text names the idea note, and the idea names the raw file.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { IDEAS, parseIdea, SENT_LINE, targetText, type Idea } from "./ideas.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const WORKSPACE_FILE = "naive-workspace.json";

export { SENT_LINE };

/** The Naive workspace at or above a folder, or null. */
export function findNaiveWorkspace(start: string): string | null {
  let dir = resolve(start);
  for (;;) {
    if (existsSync(join(dir, WORKSPACE_FILE))) return dir;
    const up = dirname(dir);
    if (up === dir) return null;
    dir = up;
  }
}

/**
 * What Naive receives: the idea in the person's words, the inputs they said fed it, and where it came from.
 * Maybe-inputs and recommendations stay in Reflect: they are suspicions and other people's suggestions, not
 * the idea.
 */
export function handoffText(idea: Idea): string {
  if (!idea.statement) throw new Error(`ideas/${idea.id}.md has no statement to send`);
  const lines = [`# ${idea.title}`, "", idea.statement];
  if (idea.from.length) lines.push("", "From:", ...idea.from.map((x) => `- ${targetText(x)}${x.how ? `: ${x.how}` : ""}`));
  lines.push("", `Sent from Reflect: ${IDEAS}/${idea.id}.md, stated ${idea.stated.at(-1)!.on}.`);
  return lines.join("\n");
}

/** Where an idea was sent before: the `sent to naive` lines on its note. */
export function sentOf(text: string): { on: string; workspace: string; raw: string }[] {
  return text
    .split("\n")
    .map((l) => (l.startsWith("- ") ? SENT_LINE.exec(l.slice(2)) : null))
    .filter((m): m is RegExpExecArray => m !== null)
    .map((m) => ({ on: m[1]!, workspace: m[2]!, raw: m[3]! }));
}

/** Runs `naive capture <text>` in the workspace and returns what it printed. */
export type CaptureRunner = (workspace: string, text: string) => { code: number; out: string; err: string };

export const runNaiveCapture: CaptureRunner = (workspace, text) => {
  const r = spawnSync(process.env.NAIVE_BIN ?? "naive", ["capture", text], { cwd: workspace, encoding: "utf8" });
  if (r.error) throw new Error(`could not run naive (${(r.error as NodeJS.ErrnoException).code ?? r.error.message}); install naive-framework or set NAIVE_BIN`);
  return { code: r.status ?? 1, out: r.stdout ?? "", err: r.stderr ?? "" };
};

export type Handoff = { idea: Idea; text: string; workspace: string; sent: { on: string; workspace: string; raw: string }[] };

/** What sending would do, without sending: the text, the workspace, and any earlier sends. */
export async function planHandoff(root: string, id: string, workspace: string | null): Promise<Handoff> {
  const rel = join(IDEAS, `${id}.md`);
  const note = await readRecord(root, rel);
  if (note === null) throw new Error(`no idea ${rel}; \`reflect idea\` lists them`);
  const ws = workspace ? resolve(workspace) : findNaiveWorkspace(root);
  if (!ws) throw new Error(`no Naive workspace found above the vault; name one with --workspace <dir> (a folder with ${WORKSPACE_FILE})`);
  if (!existsSync(join(ws, WORKSPACE_FILE))) throw new Error(`${ws} is not a Naive workspace (no ${WORKSPACE_FILE})`);
  const idea = parseIdea(id, note);
  return { idea, text: handoffText(idea), workspace: ws, sent: sentOf(note) };
}

/** Send it: run naive capture, then add a `sent to naive` line to the idea naming the raw file it kept. */
export async function sendIdea(root: string, plan: Handoff, run: CaptureRunner, now: Date): Promise<string> {
  const r = run(plan.workspace, plan.text);
  const raw = /^kept (raw\/\S+\.md)$/m.exec(r.out)?.[1];
  if (r.code !== 0 || !raw) throw new Error(`naive capture did not keep it${r.err.trim() ? `: ${r.err.trim()}` : ""}`);
  const at = now.toISOString();
  const rel = join(IDEAS, `${plan.idea.id}.md`);
  const old = (await readRecord(root, rel))!;
  const ws = basename(plan.workspace).replace(/[:\s]+/g, "-");
  await writeRecord(root, rel, `${old.replace(/\n*$/, "")}\n- ${at.slice(0, 10)} sent to naive ${ws}:${raw}\n`, "append", at);
  return `${ws}:${raw}`;
}
