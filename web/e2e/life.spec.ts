import { expect, test } from "@playwright/test";

// The apps behind the home screen's other tiles: a tile opens its app, a record is kept with a state and an act,
// and the log shows what was done. Needs open mode (next dev).
test("a friend kept from the Friends tile, with a state and a visit, shows in the log", async ({ page }, info) => {
  const name = `Sam ${info.project.name} ${Date.now()}`;
  await page.goto("/");
  await page.getByRole("link", { name: /^Friends,/ }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Friends" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("navigation", { name: "Friends parts" }).getByRole("link", { name: "Friends" }).click();
  const add = page.locator("details", { hasText: "Add friend" });
  if ((await add.getAttribute("open")) === null) await add.locator("summary").click();
  await page.getByLabel("Name").fill(name);
  await page.getByLabel("How you met").fill("At school");
  await page.getByRole("button", { name: "Keep this friend" }).click();
  await expect(page.getByRole("heading", { level: 1, name })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Close friend" }).click();
  await expect(page.getByRole("button", { name: "Close friend" })).toHaveAttribute("aria-pressed", "true", { timeout: 30_000 });
  await page.getByRole("button", { name: "Keep", exact: true }).click();
  await expect(page.getByText("Time together").last()).toBeVisible({ timeout: 30_000 });
  await page.goto("/log");
  await expect(page.getByText(`Spent time with ${name}`)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(`${name}: Close friend`)).toBeVisible();
});

test("income against the intended household on Work's income tab", async ({ page }) => {
  await page.goto("/life/work?k=income");
  await expect(page.getByText("Income coming in")).toBeVisible({ timeout: 60_000 });
  await page.goto("/life/home?k=water");
  await expect(page.getByText(/gallon per person per day/)).toBeVisible();
});

test("browsing work, a job kept as an inspiration with what draws you to it", async ({ page }, info) => {
  const draws = `Making things people use ${info.project.name} ${Date.now()}`;
  // Each project keeps a different job, so runs in parallel on the shared open-mode user do not overwrite each other.
  const job = info.project.name === "phone" ? "Data Scientists" : "Software Developers";
  await page.goto("/life/work");
  await expect(page.getByRole("heading", { name: "Forming now" })).toBeVisible({ timeout: 60_000 });
  const search = page.getByRole("search");
  await search.getByLabel("Search work").fill(job);
  await search.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Browse work" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("link", { name: new RegExp(`^Computers and mathematics ${job}`) }).first().click();
  await expect(page.getByRole("heading", { level: 1, name: job })).toBeVisible({ timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  await page.getByLabel("What draws me to it").fill(draws);
  await expect(page.getByLabel("What draws me to it")).toHaveValue(draws);
  await page.locator("label", { hasText: "Reasoning" }).click();
  // The open-mode user is shared across runs, so the job may already be kept: then the button saves it.
  const saved = page.waitForResponse((r) => r.request().method() === "POST", { timeout: 30_000 });
  await page.locator("#inspires").getByRole("button", { name: /^(It inspires me|Save)$/ }).click();
  await saved;
  await expect(page.getByRole("heading", { name: "Why it inspires you" })).toBeVisible({ timeout: 30_000 });
  await page.reload();
  await expect(page.getByLabel("What draws me to it")).toHaveValue(draws, { timeout: 30_000 });
  await page.goto("/life/work?k=inspired");
  await expect(page.getByText(draws)).toBeVisible({ timeout: 30_000 });
});
