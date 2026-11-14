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
  await page.getByLabel("Search courses from Open Yale and MIT").press("Enter");
  const card = page.locator("li", { hasText: "Death (PHIL 176)" }).first();
  await expect(card).toBeVisible({ timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  if (await card.getByRole("button", { name: "Start here" }).count()) await card.getByRole("button", { name: "Start here" }).click();
  else await card.getByRole("link", { name: "Open" }).click();
  // Started, it opens the kept course with its lectures listed; opened, the course's page lists them in its sidebar.
  await page.waitForURL(/\/learn\/(items|courses)/, { timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  if (/\/learn\/items\//.test(page.url())) await page.locator("#lectures").getByRole("link", { name: "Course Introduction" }).click({ timeout: 60_000 });
  else {
    if (!(await page.getByRole("navigation", { name: "This course" }).filter({ visible: true }).count())) await page.locator("details summary").first().click();
    await page.getByRole("navigation", { name: "This course" }).filter({ visible: true }).getByRole("link", { name: "Course Introduction" }).click({ timeout: 60_000 });
  }
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

// A course's own page, explored before keeping it: its syllabus in tabs with its texts as books, its lectures in the
// course's sidebar, and a lecture's page with its transcript closed until opened.
test("explore a course before keeping it", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto(`/learn/courses?u=${encodeURIComponent("https://oyc.yale.edu/history/hist-202")}`);
  await expect(page.getByRole("heading", { level: 1, name: /European Civilization/ })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("tab", { name: "Required texts" }).click();
  await expect(page.getByRole("tabpanel").getByRole("link", { name: "Germinal" }).first()).toBeVisible();
  if (!(await page.getByRole("navigation", { name: "This course" }).filter({ visible: true }).count())) await page.locator("details summary").first().click();
  await page.getByRole("navigation", { name: "This course" }).filter({ visible: true }).getByRole("listitem").nth(1).getByRole("link").click();
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

// Before a course: what is causing you to learn it, with its origins, and for the sake of what. It is kept in the
// journal, shown on the course, and read again before an examination.
test("write why you are learning a course", async ({ page }) => {
  test.setTimeout(240_000);
  const u = encodeURIComponent("https://oyc.yale.edu/history/hist-210");
  await page.goto(`/learn/courses/why?u=${u}&t=Early%20Middle%20Ages`);
  await expect(page.getByRole("heading", { name: "Why I am learning this" })).toBeVisible({ timeout: 60_000 });
  await page.waitForLoadState("networkidle");
  const cause = `Rome fell and I never knew what came after (${Date.now()}).`;
  await page.getByLabel("What is causing me to learn this").fill(cause);
  await page.getByLabel("Spirit", { exact: true }).check({ force: true });
  await page.getByLabel("For the sake of what").fill("To understand it.");
  await page.getByLabel("For its own sake").check({ force: true });
  await page.getByRole("button", { name: "Keep it" }).click();
  // The desktop and phone runs write the same course at once, so either one's words may be the latest.
  await expect(page.locator("#why").getByText(/Rome fell and I never knew what came after/)).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#why").getByText(/From .*spirit/)).toBeVisible();
  await page.goto(`/learn/examinations/new?u=${u}&t=Early%20Middle%20Ages`);
  await expect(page.getByText(/Rome fell and I never knew what came after/)).toBeVisible({ timeout: 60_000 });
});
