import { expect, test } from "@playwright/test";

// Browse for every medium, and a course taken inside Reflect: Browse's Courses tab starts an Open Yale course, its
// first lecture plays here with the transcript beside it, and finishing it opens the next. Needs open mode (next dev)
// and Open Yale Courses reachable.
test("browse courses, start one, and take its first lecture in Reflect", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/learn/browse");
  await expect(page.getByRole("heading", { name: "Courses to take here" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("navigation", { name: "What to browse" }).getByRole("link", { name: "Courses" }).click();
  await page.getByLabel("Search courses from Open Yale and MIT").fill("death");
  await page.getByRole("button", { name: "Search" }).click();
  const card = page.locator("li", { hasText: "Death (PHIL 176)" }).first();
  await expect(card).toBeVisible({ timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  if (await card.getByRole("button", { name: "Start here" }).count()) await card.getByRole("button", { name: "Start here" }).click();
  else await card.getByRole("link", { name: "Open" }).click();
  await expect(page.locator("#lectures")).toBeVisible({ timeout: 60_000 });
  await page.locator("#lectures").getByRole("link", { name: "Course Introduction" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Course Introduction" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: /^Chapter 1\./ })).toBeVisible();
  await page.getByLabel("Search the transcript").fill("Shelly");
  await expect(page.getByText(/call me Shelly/).first()).toBeVisible();
  const finish = page.getByRole("button", { name: "Finished, next" });
  if (await finish.count()) {
    await finish.click();
    await expect(page).toHaveURL(/\/lectures\/2$/, { timeout: 60_000 });
  }
});
