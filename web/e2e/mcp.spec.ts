import { expect, test, type APIRequestContext } from "@playwright/test";
import { createHash, randomBytes } from "node:crypto";

// A person's own AI connecting to Reflect: the app registers itself, the person signs in and allows it,
// and the app's token reaches only that person's journal through /api/mcp.
const B = "http://localhost:3000";
const REDIRECT = "http://127.0.0.1:9999/cb";
const SCOPE = "openid profile offline_access journal";

async function mcpCall(request: APIRequestContext, token: string, method: string, params: unknown) {
  const r = await request.post(`${B}/api/mcp`, {
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream", authorization: `Bearer ${token}`, "mcp-protocol-version": "2025-06-18" },
    data: { jsonrpc: "2.0", id: 1, method, params },
  });
  const body = await r.text();
  const json = body.startsWith("{") ? body : (body.match(/^data: (.*)$/m)?.[1] ?? "{}");
  return { status: r.status(), msg: JSON.parse(json) as { result?: { content?: { text: string }[]; isError?: boolean; tools?: { name: string }[] } } };
}

test("a person's AI connects, writes an entry and keeps only a reading anchored in it", async ({ page, request }, info) => {
  test.skip(info.project.name !== "desktop", "one browser is enough for the OAuth flow");

  const unauth = await request.post(`${B}/api/mcp`, { data: {} });
  expect(unauth.status()).toBe(401);
  expect(unauth.headers()["www-authenticate"]).toContain("/.well-known/oauth-protected-resource/api/mcp");
  const meta = await (await request.get(`${B}/.well-known/oauth-protected-resource/api/mcp`)).json();
  expect(meta.resource).toBe(`${B}/api/mcp`);

  const reg = await (await request.post(`${B}/api/auth/oauth2/register`, {
    data: { client_name: "Test AI", application_type: "native", redirect_uris: [REDIRECT], token_endpoint_auth_method: "none", grant_types: ["authorization_code", "refresh_token"], response_types: ["code"], scope: SCOPE },
  })).json();
  expect(reg.client_id).toBeTruthy();

  const verifier = randomBytes(32).toString("base64url");
  const q = new URLSearchParams({
    response_type: "code", client_id: reg.client_id, redirect_uri: REDIRECT, scope: SCOPE, state: "s1",
    code_challenge: createHash("sha256").update(verifier).digest("base64url"), code_challenge_method: "S256", resource: `${B}/api/mcp`,
  });
  let code: string | null = null;
  await page.route("http://127.0.0.1:9999/**", (r) => {
    code = new URL(r.request().url()).searchParams.get("code");
    return r.fulfill({ body: "done" });
  });
  await page.goto(`${B}/api/auth/oauth2/authorize?${q}`);
  await page.waitForLoadState("networkidle");
  await page.fill("#email", `mcp-${Date.now()}@example.test`);
  await page.fill("#password", "correct-horse-battery");
  await page.click("button[type=submit]");
  await expect(page.getByRole("heading", { name: "Connect an app to your journal?" })).toBeVisible();
  await expect(page.getByText("calls itself “Test AI”")).toBeVisible();
  await page.getByRole("button", { name: "Allow", exact: true }).click();
  await expect.poll(() => code).toBeTruthy();

  const tok = await (await request.post(`${B}/api/auth/oauth2/token`, {
    form: { grant_type: "authorization_code", code: code!, redirect_uri: REDIRECT, client_id: reg.client_id, code_verifier: verifier, resource: `${B}/api/mcp` },
  })).json();
  expect(tok.access_token).toBeTruthy();

  const tools = await mcpCall(request, tok.access_token, "tools/list", {});
  expect(tools.msg.result?.tools?.map((t) => t.name)).toEqual(["reflect_guide", "list_entries", "write_entry", "get_entry", "keep_reading", "concept_history", "keep_legacy_plan", "legacy_status", "learning", "add_book", "add_book_subjects", "set_book_status", "recall_check", "add_courses", "add_credential", "rate_subject", "add_input", "mark_subject", "ideas", "keep_idea"]);

  const today = new Date().toISOString().slice(0, 10);
  const wrote = await mcpCall(request, tok.access_token, "tools/call", { name: "write_entry", arguments: { text: "Dad called. We laughed about the lake.", local_date: today } });
  const id = JSON.parse(wrote.msg.result!.content![0]!.text).id as string;

  const kept = await mcpCall(request, tok.access_token, "tools/call", {
    name: "keep_reading",
    arguments: { entry_id: id, bindings: [{ element: "emotion", weight: 0.8, on: [{ concept: { kind: "entity", name: "Dad", sort: "person" }, words: "Dad", role: "agent" }, { concept: { kind: "event", name: "laugh" }, words: "laughed" }] }] },
  });
  expect(kept.msg.result?.isError).toBeFalsy();

  const refused = await mcpCall(request, tok.access_token, "tools/call", {
    name: "keep_reading",
    arguments: { entry_id: id, bindings: [{ element: "emotion", weight: 0.5, on: [{ concept: { kind: "quality", name: "sad" }, words: "sad" }] }] },
  });
  expect(refused.msg.result?.isError).toBe(true);
  expect(refused.msg.result?.content?.[0]?.text).toContain("not in the entry");

  const got = await mcpCall(request, tok.access_token, "tools/call", { name: "get_entry", arguments: { entry_id: id } });
  const entry = JSON.parse(got.msg.result!.content![0]!.text) as { readings: { bindings: unknown[] }[] };
  expect(entry.readings).toHaveLength(1);

  // The person's AI keeps their legacy plan; a trustee confirms; the recipient then sees their letter.
  const legacy = await mcpCall(request, tok.access_token, "tools/call", {
    name: "keep_legacy_plan",
    arguments: { confirmations: 1, wait_days: 0, trustees: [{ name: "Ana", contact: "ana@example.test" }], gifts: [{ to: "Sam", contact: "sam@example.test", items: [{ title: "Dear Sam", text: "Stay brave." }] }] },
  });
  expect(legacy.msg.result?.isError).toBeFalsy();
  const trusteeLink = (JSON.parse(legacy.msg.result!.content![0]!.text) as { trustees: { link: string }[] }).trustees[0]!.link;
  await page.goto(new URL(trusteeLink).pathname);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("named you, Ana, a trustee");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Confirm", exact: true }).click();
  await expect(page.getByText("The release happened on")).toBeVisible();
  const receiveLink = await page.locator("code").first().textContent();
  await page.goto(new URL(receiveLink!).pathname);
  await expect(page.getByRole("heading", { name: "Dear Sam" })).toBeVisible();
  await expect(page.getByText("Stay brave.")).toBeVisible();
});
