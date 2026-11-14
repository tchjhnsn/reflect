import { expect, test } from "@playwright/test";

// The calendar, a tool: an event tagged with a tile shows in the master calendar and in that tile's calendar only.
// Needs open mode (next dev).
test("a weekly shift tagged Work shows in the master calendar and Work's, not Learn's", async ({ page }, info) => {
  const title = `Shift ${info.project.name} ${Date.now()}`;
  const day = new Date().toISOString().slice(0, 10);
  await page.goto("/life/work");
  await page.getByRole("link", { name: "Work's calendar" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Work calendar" })).toBeVisible({ timeout: 60_000 });
  const add = page.locator("details", { hasText: "Add an event" }).last();
  if ((await add.getAttribute("open")) === null) await add.locator("summary").click();
  await add.getByLabel("What").fill(title);
  await add.getByLabel("Day", { exact: true }).fill(day);
  await add.getByLabel("Starts").fill("09:00");
  await add.getByLabel("Repeats").selectOption("weekly");
  await add.getByRole("button", { name: "Add to the calendar" }).click();
  await expect(page.getByText(title).last()).toBeVisible({ timeout: 30_000 });
  await page.goto(`/calendar?day=${day}`);
  await expect(page.getByText(title).last()).toBeVisible({ timeout: 30_000 });
  await page.goto(`/calendar?tile=learn&day=${day}`);
  await expect(page.getByRole("heading", { level: 1, name: "Learn calendar" })).toBeVisible();
  await expect(page.getByText(title)).toHaveCount(0);
});
