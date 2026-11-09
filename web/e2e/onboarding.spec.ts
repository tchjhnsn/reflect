import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { expect, test, type Page } from "@playwright/test";

const shots = process.env.SHOTS_DIR;
const axeSource = readFileSync(createRequire(import.meta.url).resolve("axe-core/axe.min.js"), "utf8");

async function settle(page: Page) {
  await page.evaluate(() => Promise.all(document.getAnimations().map((a) => a.finished)));
}

async function shot(page: Page, name: string) {
  await settle(page);
  if (shots) await page.screenshot({ path: `${shots}/${test.info().project.name}-${name}.png`, fullPage: true });
}

async function answer(page: Page, prompt: string, text: string) {
  await expect(page.getByLabel(prompt)).toBeVisible();
  await page.getByLabel(prompt).fill(text);
  await page.getByRole("button", { name: "Next" }).click();
}

async function seriousAxe(page: Page) {
  await settle(page);
  await page.addScriptTag({ content: axeSource });
  const result = await page.evaluate(async () => {
    const r = await (window as unknown as { axe: { run: () => Promise<{ violations: { id: string; impact: string }[] }> } }).axe.run();
    return r.violations
      .filter((v) => v.impact === "serious" || v.impact === "critical")
      .map((v) => `${v.id}: ${(v as unknown as { nodes: { target: string[] }[] }).nodes.map((n) => n.target.join(" ")).join(", ")}`);
  });
  expect(result).toEqual([]);
}

test("a full run reaches the review with five cards and keeps them", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (r) => requests.push(`${r.url()} ${r.postData() ?? ""}`));
  await page.goto("/welcome");
  await shot(page, "1-name");
  await seriousAxe(page);
  await answer(page, "What should we call you?", "Ada");
  await shot(page, "2-born");
  await answer(page, "When were you born?", "1990-04-01");
  await answer(page, "What do you do?", "Engineer");
  await shot(page, "4-friend");
  await answer(page, "Who is your closest friend?", "Sam");
  await shot(page, "5-puzzle");
  await answer(page, "What is the puzzle of your life right now?", "Why do I keep starting over?");
  await expect(page.getByRole("heading", { name: "Here’s what you told us." })).toBeVisible();
  for (const id of ["name", "date_of_birth", "occupation", "closest_friend", "puzzle"]) {
    await expect(page.getByTestId(`card-${id}`)).toBeVisible();
  }
  await expect(page.getByTestId("card-puzzle")).toContainText("Why do I keep starting over?");
  await shot(page, "6-review");
  await seriousAxe(page);
  await page.getByRole("button", { name: "Keep these" }).click();
  await expect(page.getByTestId("saved-count")).toHaveText("Kept 5 of your answers.");
  await shot(page, "7-done");
  for (const secret of ["Ada", "Sam", "starting over"]) {
    expect(requests.some((r) => r.includes(secret))).toBe(false);
  }
});

test("skipping everything keeps nothing", async ({ page }) => {
  await page.goto("/welcome");
  for (let i = 0; i < 5; i++) await page.getByRole("button", { name: "Skip" }).click();
  await expect(page.getByTestId("nothing")).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByTestId("saved-count")).toHaveText("Nothing was kept. You can add any of it later.");
});

test("edit and remove on the review", async ({ page }) => {
  await page.goto("/welcome");
  await answer(page, "What should we call you?", "Ada");
  await answer(page, "When were you born?", "1990-04-01");
  await answer(page, "What do you do?", "Engineer");
  await answer(page, "Who is your closest friend?", "Sam");
  await page.getByRole("button", { name: "Skip" }).click();
  await page.getByTestId("card-occupation").getByRole("button", { name: "Edit" }).click();
  await expect(page.getByLabel("What do you do?")).toHaveValue("Engineer");
  await page.getByLabel("What do you do?").fill("Teacher");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByTestId("card-occupation")).toContainText("Teacher");
  await expect(page.getByTestId("card-closest_friend")).toContainText("Sam");
  await page.getByRole("button", { name: "Remove Closest friend" }).click();
  await expect(page.getByTestId("card-closest_friend")).toHaveCount(0);
  await page.getByRole("button", { name: "Keep these" }).click();
  await expect(page.getByTestId("saved-count")).toHaveText("Kept 3 of your answers.");
});

test("a future date of birth is refused in place", async ({ page }) => {
  await page.goto("/welcome");
  await answer(page, "What should we call you?", "Ada");
  await page.getByLabel("When were you born?").fill("2099-01-01");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.locator("#answer-date_of_birth-error")).toHaveText("A date of birth can't be in the future.");
  await expect(page.getByLabel("When were you born?")).toBeVisible();
  await shot(page, "error");
});

test("the whole flow works from the keyboard", async ({ page }) => {
  await page.goto("/welcome");
  await page.keyboard.type("Ada");
  await page.keyboard.press("Enter");
  await page.keyboard.type("1990-04-01".replaceAll("-", ""));
  await page.keyboard.press("Enter");
  await page.keyboard.type("Engineer");
  await page.keyboard.press("Enter");
  await page.keyboard.type("Sam");
  await page.keyboard.press("Enter");
  await page.keyboard.type("What is enough?");
  await page.keyboard.press("Control+Enter");
  await expect(page.getByTestId("card-puzzle")).toContainText("What is enough?");
});
