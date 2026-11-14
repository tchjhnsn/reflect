import { expect, test } from "@playwright/test";

// Learn's shelves by the ways of learning: a game kept for rest moves to Practice once the person says what it
// taught them; its page keeps practice sessions; a removal can be undone. Needs open mode (next dev).
test("a game moves from rest to practice with a lesson, keeps sessions, and a removal is undone", async ({ page }, info) => {
  test.setTimeout(180_000);
  const title = `Go ${info.project.name} ${Date.now()}`;
  await page.goto("/learn?filled=1");
  await page.getByLabel("Medium").first().selectOption("game");
  await page.locator("form input[name=title]").fill(title);
  await page.locator("form").getByRole("button", { name: "Add", exact: true }).click();
  const leisure = page.locator("details#leisure");
  await expect(leisure.getByText(title)).toBeVisible({ timeout: 30_000 });

  await leisure.getByText(title).click();
  await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible({ timeout: 30_000 });
  await page.getByLabel("What it taught you").fill("Influence over territory: a stone's worth is where it stands.");
  await page.getByRole("button", { name: "Keep", exact: true }).click();
  await expect(page.getByText("Influence over territory")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("My shelves · Practice")).toBeVisible();

  await page.getByLabel("Minutes").fill("30");
  await page.getByRole("button", { name: "Keep session" }).click();
  await expect(page.getByText(/1 session in the last 30 days/)).toBeVisible({ timeout: 30_000 });

  await page.getByRole("button", { name: "Remove from your shelves" }).click();
  await expect(page.getByText(`Removed ${title}.`)).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible({ timeout: 30_000 });
});
