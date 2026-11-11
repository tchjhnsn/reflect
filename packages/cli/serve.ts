// `reflect serve`: screens for the soul vault, served from this machine only. The soul vault is local files
// that no hosted service reaches, so these pages are rendered here, from the vault, by the same core code the
// commands use, and every change goes through it too (bind, unbind, hold, confirm), logged in memory.jsonl.
// The server listens on 127.0.0.1, answers only requests addressed to that host, and refuses a form posted
// from any other origin, so a web page elsewhere cannot read the vault or write to it.
import { createServer, type Server } from "node:http";
import { filesJournal } from "../core/src/files-journal.ts";
import { bindingsOf, bindMoment, boundNotes, confirmedElements, elementTotals, HELD_AS, HOLD_KINDS, keepHold, listHolds, momentFiles, noteProfile, ofMeReadings, unbindMoment } from "../core/src/ground.ts";
import { journalHome } from "../core/src/home.ts";
import { readConfig } from "../core/src/instance.ts";
import { ROLES } from "../core/src/bindings.ts";
import { CATEGORIES, elementName } from "../core/src/state.ts";
import { confirmReading } from "../core/src/wiki.ts";

export type Reply = { status: number; headers: Record<string, string>; body: string };

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const enc = encodeURIComponent;

const CSS = `
:root { --paper:#f6f1e7; --deep:#eee5d3; --ink:#24211c; --soft:#6f675b; --rule:#e0d6c3; --accent:#9a4b2b; --error:#a12d1c; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --paper:#1b1915; --deep:#24211b; --ink:#ece5d8; --soft:#a69d8e; --rule:#34302a; --accent:#e0946f; --error:#f08b7a; color-scheme: dark; } }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--paper); color: var(--ink); }
body { font: 16px/1.55 ui-sans-serif, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; text-underline-offset: 3px; }
main { max-width: 44rem; margin: 0 auto; padding: 32px 16px 64px; }
nav { display: flex; flex-wrap: wrap; gap: 4px 16px; padding-bottom: 16px; border-bottom: 1px solid var(--rule); margin-bottom: 28px; font: 500 14px ui-sans-serif, system-ui, sans-serif; }
nav a { color: var(--soft); } nav a.on { color: var(--ink); }
h1 { font: 400 34px/1.15 "Iowan Old Style", Georgia, serif; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font: 500 12px/1.4 ui-sans-serif, system-ui, sans-serif; letter-spacing: 0.12em; text-transform: uppercase; color: var(--soft); margin: 32px 0 10px; }
.note { color: var(--soft); font-size: 15px; margin: 0 0 12px; }
.serif { font-family: "Iowan Old Style", Georgia, serif; font-size: 19px; }
.text { white-space: pre-wrap; font: 19px/1.6 "Iowan Old Style", Georgia, serif; }
ul.rows { list-style: none; padding: 0; margin: 0; border-top: 1px solid var(--ink); }
ul.rows li { padding: 12px 0; border-bottom: 1px solid var(--rule); }
.meta { color: var(--soft); font-size: 13px; }
.chip { display: inline-block; font-size: 12px; padding: 1px 8px; margin: 2px 4px 2px 0; border: 1px solid var(--rule); border-radius: 999px; color: var(--ink); }
.bar { display: grid; grid-template-columns: 11rem 1fr 2.5rem; gap: 8px; align-items: center; font-size: 14px; margin: 4px 0; }
.track { height: 8px; background: var(--deep); border-radius: 4px; overflow: hidden; } .fill { height: 100%; background: var(--accent); }
@media (max-width: 30rem) { .bar { grid-template-columns: 8.5rem 1fr 2.5rem; } }
.pair { margin-bottom: 14px; } .pair .name { font: 500 12px ui-sans-serif, system-ui, sans-serif; letter-spacing: 0.1em; text-transform: uppercase; color: var(--soft); }
form.inline { display: inline; }
form.stack { display: grid; gap: 10px; margin-top: 8px; }
input[type=text], textarea, select { font: inherit; color: var(--ink); background: transparent; border: 0; border-bottom: 1px solid var(--ink); padding: 8px 0; border-radius: 0; width: 100%; }
textarea { font: 19px/1.5 "Iowan Old Style", Georgia, serif; min-height: 5rem; resize: vertical; }
select { width: auto; }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }
button { font: 500 14px ui-sans-serif, system-ui, sans-serif; cursor: pointer; border: 0; background: var(--ink); color: var(--paper); padding: 8px 16px; border-radius: 999px; }
button.quiet { background: none; color: var(--soft); padding: 2px 6px; } button.quiet:hover { color: var(--ink); }
.ok { color: var(--soft); font-size: 13px; } .bad { color: var(--error); font-size: 14px; }
.flash { padding: 10px 12px; background: var(--deep); border-radius: 8px; margin-bottom: 20px; font-size: 14px; }
`;

const NAV: [string, string][] = [["/", "Soul"], ["/moments", "Moments"], ["/notes", "Life notes"], ["/holds", "Holds"], ["/of-me", "Of me"]];

function page(title: string, at: string, body: string, flash?: string): Reply {
  const nav = NAV.map(([href, name]) => `<a href="${href}"${href === at ? ' class="on" aria-current="page"' : ""}>${name}</a>`).join("");
  return {
    status: 200,
    headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "x-frame-options": "DENY", "content-security-policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'" },
    body: `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)} · Soul</title><style>${CSS}</style></head><body><main><nav aria-label="Soul vault">${nav}</nav>${flash ? `<p class="flash" role="status">${esc(flash)}</p>` : ""}${body}</main></body></html>`,
  };
}

const redirect = (to: string): Reply => ({ status: 303, headers: { location: to }, body: "" });
const notFound = (): Reply => ({ status: 404, headers: { "content-type": "text/plain; charset=utf-8" }, body: "not found" });
const chips = (els: readonly string[]) => (els.length ? els.map((e) => `<span class="chip">${esc(elementName(e))}</span>`).join("") : '<span class="meta">no confirmed reading</span>');
const back = (to: string, msg: string) => redirect(`${to}${to.includes("?") ? "&" : "?"}msg=${enc(msg)}`);

/** The eight categories, each pair with its counts as bars against the largest. */
function pairs(elements: Record<string, number>, of: number): string {
  const max = Math.max(1, ...Object.values(elements));
  return CATEGORIES.map((c) => `<div class="pair"><div class="name">${esc(c.id)}</div>${c.pair.map((e) => {
    const n = elements[e] ?? 0;
    return `<div class="bar"><span>${esc(elementName(e))}</span><span class="track"><span class="fill" style="display:block;width:${Math.round((100 * n) / max)}%"></span></span><span class="meta">${n}${of ? `/${of}` : ""}</span></div>`;
  }).join("")}</div>`).join("");
}

async function soulPage(root: string, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return page("No soul vault", "/", `<h1>No soul vault yet</h1><p class="note">Your journal and readings can live in a vault of their own, apart from your life vault. Start one with <code>reflect soul init &lt;folder&gt;</code>, then open this page again.</p>`);
  const t = await elementTotals(root);
  const notes = await boundNotes(root);
  const holds = await listHolds(root);
  const readings = await ofMeReadings(root);
  return page("Soul", "/", `<h1>Your soul vault</h1>
<p class="note">${esc(home.root)}</p>
<p class="serif">${t.moments} moment${t.moments === 1 ? "" : "s"}, ${t.read} with a confirmed reading. ${notes.length} life note${notes.length === 1 ? "" : "s"} bound. ${holds.length} hold${holds.length === 1 ? "" : "s"}. ${readings.length} reading${readings.length === 1 ? "" : "s"} of you, ${readings.filter((r) => r.confirmed && !r.problem).length} confirmed.</p>
<h2>Elements across your moments</h2>
<p class="note">How many moments show each element in what you confirmed. Model guesses do not count.</p>
${pairs(t.elements, t.read)}`, msg);
}

async function momentsPage(root: string, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return soulPage(root);
  const config = await readConfig(home.root);
  const entries = await filesJournal(home.root, config.person, { layout: home.layout, lifeRoot: home.lifeRoot }).listEntries(10_000);
  const rows: string[] = [];
  for (const e of entries) {
    const bound = await bindingsOf(root, e.id);
    rows.push(`<li><a class="serif" href="/moments/${enc(e.id)}">${esc((e.parts[0]?.text ?? "").split("\n")[0]!.slice(0, 90) || "(empty)")}</a><div class="meta">${esc(e.localDate)}${bound.length ? ` · about ${bound.map((b) => esc(b.to)).join(", ")}` : ""}</div><div>${chips(await confirmedElements(home, e.id))}</div></li>`);
  }
  return page("Moments", "/moments", `<h1>Moments</h1><p class="note">Your journal. Open a moment to say what it was about.</p><ul class="rows">${rows.join("") || "<li class=\"meta\">No moments yet.</li>"}</ul>`, msg);
}

async function momentPage(root: string, id: string, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return soulPage(root);
  const m = (await momentFiles(home)).find((x) => x.id === id);
  if (!m) return notFound();
  const config = await readConfig(home.root);
  const journal = filesJournal(home.root, config.person, { layout: home.layout, lifeRoot: home.lifeRoot });
  const entry = (await journal.listEntries(10_000)).find((e) => e.id === id);
  const text = await journal.entryText(id);
  const bound = await bindingsOf(root, id);
  const here = `/moments/${enc(id)}`;
  const roles = ROLES.map((r) => `<option value="${r}">${r}</option>`).join("");
  return page(entry?.localDate ?? id, "/moments", `<p class="meta"><a href="/moments">Moments</a> · ${esc(entry?.localDate ?? "")}</p>
<div class="text">${esc(text ?? "")}</div>
<h2>Confirmed elements</h2><div>${chips(await confirmedElements(home, id))}</div>
<p class="note">Reflect on a moment and keep what you confirm with <code>reflect read</code> and <code>reflect keep</code>.</p>
<h2>What it was about</h2>
<ul class="rows">${bound.map((b) => `<li><a href="/notes?note=${enc(b.to)}">${esc(b.to)}</a>${b.role ? ` <span class="meta">(${esc(b.role)})</span>` : ""} <form class="inline" method="post" action="/unbind"><input type="hidden" name="moment" value="${esc(id)}"><input type="hidden" name="note" value="${esc(b.to)}"><input type="hidden" name="back" value="${esc(here)}"><button class="quiet" type="submit">unbind</button></form></li>`).join("") || '<li class="meta">Bound to nothing yet.</li>'}</ul>
<form class="stack" method="post" action="/bind"><input type="hidden" name="moment" value="${esc(id)}"><input type="hidden" name="back" value="${esc(here)}">
<label for="note" class="meta">A note or folder in your life vault, by its path (work/bakery, belonging/friends/sam)</label>
<div class="row"><input id="note" type="text" name="note" required autocomplete="off" style="flex:1;min-width:12rem"><select name="role" aria-label="Role"><option value="">no role</option>${roles}</select><button type="submit">Bind</button></div></form>
<p class="note">The life note is never changed. Bindings live here, in the soul vault.</p>`, msg);
}

async function notesPage(root: string, note: string | null, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return soulPage(root);
  if (note) {
    const p = await noteProfile(root, note);
    return page(p.note, "/notes", `<p class="meta"><a href="/notes">Life notes</a></p><h1>${esc(p.note)}</h1>
<p class="note">${p.moments.length} moment${p.moments.length === 1 ? "" : "s"} bound${p.unread ? `, ${p.unread} with no confirmed reading yet` : ""}. Counted now from what you confirmed; nothing here is stored.</p>
${p.elements.length ? pairs(Object.fromEntries(p.elements.map((e) => [e.element, e.moments])), p.moments.length - p.unread) : '<p class="note">No confirmed elements yet.</p>'}
<h2>Moments</h2><ul class="rows">${p.moments.map((id) => `<li><a href="/moments/${enc(id)}">${esc(id)}</a></li>`).join("") || '<li class="meta">None.</li>'}</ul>`, msg);
  }
  const notes = await boundNotes(root);
  return page("Life notes", "/notes", `<h1>Life notes</h1><p class="note">The notes in your life vault your moments are bound to. Open one to see the elements you confirmed on those moments.</p>
<ul class="rows">${notes.map((n) => `<li><a class="serif" href="/notes?note=${enc(n.note)}">${esc(n.note)}</a> <span class="meta">${n.moments} moment${n.moments === 1 ? "" : "s"}</span></li>`).join("") || '<li class="meta">No bindings yet. Open a moment to bind it.</li>'}</ul>`, msg);
}

async function holdsPage(root: string, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return soulPage(root);
  const holds = await listHolds(root);
  return page("Holds", "/holds", `<h1>Holds</h1><p class="note">Beliefs and values in your own words, and how you hold them. Never inferred. Stating one again under the same name adds a line.</p>
<ul class="rows">${holds.map((h) => `<li><div class="serif">${esc(h.words)}</div><div class="meta">${esc(h.kind)}${h.held ? `, held as ${esc(h.held)} (${esc(elementName(h.element ?? ""))})` : ""} · ${esc(h.on)}${h.stated > 1 ? ` · stated ${h.stated} times` : ""} · holds/${esc(h.id)}.md</div></li>`).join("") || '<li class="meta">No holds yet.</li>'}</ul>
<h2>State one</h2>
<form class="stack" method="post" action="/hold">
<label for="words" class="meta">In your words</label><textarea id="words" name="words" required></textarea>
<div class="row"><select name="as" aria-label="Kind">${HOLD_KINDS.map((k) => `<option value="${k}">${k}</option>`).join("")}</select>
<select name="held" aria-label="Held as"><option value="">held as…</option>${HELD_AS.map((h) => `<option value="${h}">${h}</option>`).join("")}</select>
<input type="text" name="name" placeholder="name (optional)" aria-label="Name" style="flex:1;min-width:10rem"><button type="submit">Keep</button></div></form>`, msg);
}

async function ofMePage(root: string, msg?: string): Promise<Reply> {
  const home = await journalHome(root);
  if (!home.soul) return soulPage(root);
  const rs = await ofMeReadings(root);
  const link = (body: string) => esc(body).replace(/\[\[(moments\/[^\]|#]+)(?:[|#][^\]]*)?\]\]/g, (_m, p: string) => {
    const stem = p.slice("moments/".length);
    return `<a href="/moment-file/${enc(stem)}">${esc(stem.slice(0, 10))}</a>`;
  });
  return page("Of me", "/of-me", `<h1>Readings of you</h1><p class="note">Patterns, tensions and identities your AI read in your moments. Each names its elements and cites moments whose confirmed readings show them. A reading stays the AI's until you confirm it.</p>
<ul class="rows">${rs.map((r) => `<li><div class="serif">${esc(r.title)}</div><div class="meta">${esc(r.type)} · ${r.elements.map((e) => esc(elementName(e))).join(r.type === "tension" ? " against " : ", ")}</div>
<p class="text" style="font-size:16px">${link(r.body)}</p>
${r.problem ? `<p class="bad">Does not answer to the element table: ${esc(r.problem)}</p>` : r.confirmed ? `<p class="ok">Confirmed ${esc(r.confirmed)}</p>` : `<form method="post" action="/confirm"><input type="hidden" name="path" value="${esc(r.path)}"><button type="submit">Confirm as mine</button></form>`}</li>`).join("") || '<li class="meta">None yet. They come from the wiki: <code>reflect wiki prompt</code>, then <code>reflect wiki absorb</code>.</li>'}</ul>`, msg);
}

/** Answer one request. Pure apart from the vault: the server below and the tests both call it. */
export async function handle(root: string, method: string, url: URL, form: URLSearchParams): Promise<Reply> {
  const msg = url.searchParams.get("msg") ?? undefined;
  const p = decodeURIComponent(url.pathname);
  const now = new Date();
  try {
    if (method === "GET") {
      if (p === "/") return await soulPage(root, msg);
      if (p === "/moments") return await momentsPage(root, msg);
      if (p.startsWith("/moments/")) return await momentPage(root, p.slice("/moments/".length), msg);
      if (p.startsWith("/moment-file/")) {
        const home = await journalHome(root);
        const m = (await momentFiles(home)).find((x) => x.stem === p.slice("/moment-file/".length));
        return m ? redirect(`/moments/${enc(m.id)}`) : notFound();
      }
      if (p === "/notes") return await notesPage(root, url.searchParams.get("note"), msg);
      if (p === "/holds") return await holdsPage(root, msg);
      if (p === "/of-me") return await ofMePage(root, msg);
      return notFound();
    }
    if (method === "POST") {
      const backTo = (form.get("back") ?? "/").startsWith("/") && !(form.get("back") ?? "").startsWith("//") ? form.get("back")! : "/";
      if (p === "/bind") {
        const r = await bindMoment(root, form.get("moment") ?? "", form.get("note") ?? "", form.get("role") || undefined, now);
        return back(backTo, r.changed ? `Bound to ${r.to}.` : `Already bound to ${r.to}.`);
      }
      if (p === "/unbind") {
        const ok = await unbindMoment(root, form.get("moment") ?? "", form.get("note") ?? "", now);
        return back(backTo, ok ? `Unbound from ${form.get("note")}.` : "It was not bound there.");
      }
      if (p === "/hold") {
        const h = await keepHold(root, form.get("words") ?? "", form.get("as") ?? "", form.get("held") || undefined, form.get("name") || undefined, now);
        return back("/holds", `Kept holds/${h.id}.md.`);
      }
      if (p === "/confirm") {
        await confirmReading(root, form.get("path") ?? "", now);
        return back("/of-me", "Confirmed as yours.");
      }
      return notFound();
    }
    return { status: 405, headers: { allow: "GET, POST" }, body: "" };
  } catch (e) {
    const why = (e as Error).message;
    if (method === "POST") return back(p === "/hold" ? "/holds" : p === "/confirm" ? "/of-me" : (form.get("back") ?? "/").startsWith("/") ? form.get("back")! : "/", `Not kept: ${why}`);
    return { status: 500, headers: { "content-type": "text/plain; charset=utf-8" }, body: why };
  }
}

/** Whether a request may be answered: addressed to this loopback server, and a form posted from these pages. */
export function allowed(method: string, host: string | undefined, origin: string | undefined, port: number): boolean {
  const hosts = [`127.0.0.1:${port}`, `localhost:${port}`];
  if (!host || !hosts.includes(host)) return false;
  if (method === "POST" && origin !== undefined && !hosts.map((h) => `http://${h}`).includes(origin)) return false;
  return true;
}

/** Serve the soul vault's screens on 127.0.0.1. */
export function serve(root: string, port: number): Promise<Server> {
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 64 * 1024) req.destroy();
      else chunks.push(c);
    });
    req.on("end", () => {
      const method = req.method ?? "GET";
      if (!allowed(method, req.headers.host, req.headers.origin, port)) {
        res.writeHead(403, { "content-type": "text/plain; charset=utf-8" }).end("forbidden");
        return;
      }
      const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
      const form = new URLSearchParams(method === "POST" ? Buffer.concat(chunks).toString("utf8") : "");
      handle(root, method, url, form).then(
        (r) => res.writeHead(r.status, r.headers).end(r.body),
        (e: Error) => res.writeHead(500, { "content-type": "text/plain; charset=utf-8" }).end(e.message),
      );
    });
  });
  return new Promise((ok, fail) => {
    server.once("error", fail);
    server.listen(port, "127.0.0.1", () => ok(server));
  });
}
