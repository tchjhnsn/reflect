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

test("income against the intended household on Work's overview", async ({ page }) => {
  await page.goto("/life/work");
  await expect(page.getByText("Income coming in")).toBeVisible({ timeout: 60_000 });
  await page.goto("/life/home?k=water");
  await expect(page.getByText(/gallon per person per day/)).toBeVisible();
});
