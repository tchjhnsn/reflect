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
  await page.getByRole("button", { name: "Transcript" }).click();
  await expect(page.getByRole("button", { name: /^Chapter 1\./ })).toBeVisible();
  await page.getByLabel("Search the transcript").fill("Shelly");
  await expect(page.getByText(/call me Shelly/).first()).toBeVisible();
  const finish = page.getByRole("button", { name: "Finished, next" });
  if (await finish.count()) {
    await finish.click();
    await expect(page).toHaveURL(/\/lectures\/2$/, { timeout: 60_000 });
  }
});

// A course's own page, explored before keeping it: its syllabus in tabs with its texts as books, a lecture played in
// place from the list, and the lecture's page with its transcript closed until opened.
test("explore a course before keeping it", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto(`/learn/courses?u=${encodeURIComponent("https://oyc.yale.edu/history/hist-202")}`);
  await expect(page.getByRole("heading", { level: 1, name: /European Civilization/ })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("tab", { name: "Required texts" }).click();
  await expect(page.getByRole("tabpanel").getByRole("link", { name: "Germinal" }).first()).toBeVisible();
  await page.locator("#lectures li").nth(1).getByRole("button").first().click();
  await expect(page.locator("#lectures iframe")).toBeVisible({ timeout: 60_000 });
  await page.locator("#lectures li").nth(1).getByRole("link", { name: "Open" }).click();
  await expect(page.getByLabel("Search the transcript")).toHaveCount(0, { timeout: 60_000 });
  await page.getByRole("button", { name: "Transcript" }).click();
  await expect(page.getByRole("button", { name: /^Chapter 1\./ })).toBeVisible();
});

// Putting yourself under examination on a course: what you answer for, then the link to invite examiners. Examiners
// joining and attesting are covered by the store's test, since open mode has one person.
test("put yourself under examination on a course", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto(`/learn/courses?u=${encodeURIComponent("https://oyc.yale.edu/philosophy/phil-176")}&t=Death`);
  await page.getByRole("link", { name: "Put yourself under examination" }).click();
  await expect(page.getByRole("heading", { name: "Put yourself under examination" })).toBeVisible({ timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  await page.getByLabel("Your name, as your examiners will see it").fill("Test Candidate");
  await page.getByLabel("What you will answer for").fill("Why death is bad for the one who dies.");
  await page.getByRole("button", { name: "Put yourself under examination" }).click();
  await expect(page.getByRole("heading", { name: "Your examination" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByLabel("Invitation link")).toHaveValue(/\/learn\/examine\//);
  await expect(page.getByText("Why death is bad for the one who dies.")).toBeVisible();
  const link = await page.getByLabel("Invitation link").inputValue();
  await page.goto(link);
  await expect(page.getByRole("heading", { name: "Your examination" })).toBeVisible({ timeout: 60_000 });
});
