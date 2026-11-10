import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { expect, test, type Page } from "@playwright/test";
import { signUp } from "./helpers";

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

async function pick(page: Page, prompt: string, options: string[], words?: string) {
  await expect(page.getByRole("heading", { name: prompt })).toBeVisible();
  for (const o of options) await page.getByRole(options.length === 1 && !words ? "radio" : "button", { name: o, exact: true }).click();
  if (words) await page.getByLabel("In your own words").fill(words);
  await page.getByRole("button", { name: "Next" }).click();
}

async function skipRest(page: Page) {
  while (!(await page.getByRole("heading", { name: "Here’s what you told us." }).isVisible())) await page.getByRole("button", { name: "Skip" }).click();
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

test("a full run reaches the review with every card and keeps them", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (r) => requests.push(`${r.url()} ${r.postData() ?? ""}`));
  await signUp(page);
  await shot(page, "1-name");
  await seriousAxe(page);
  await answer(page, "What should we call you?", "Ada");
  await shot(page, "2-born");
  await answer(page, "When were you born?", "1990-04-01");
  await answer(page, "What do you do?", "Engineer");
  await shot(page, "4-friend");
  await answer(page, "Who is your closest friend?", "Sam");
  await pick(page, "Are you married, or with someone?", ["Married"]);
  await pick(page, "Are you seeing anyone right now?", ["No"]);
  await answer(page, "Do you have children? How many?", "2");
  await shot(page, "5-hold");
  await seriousAxe(page);
  await pick(page, "How do you hold marriage or partnership?", ["I want it", "I feel I ought to"], "It matters to my family.");
  await page.getByRole("button", { name: "Not for me", exact: true }).click();
  await expect(page.getByRole("button", { name: "Not for me", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Unsure", exact: true }).click();
  await expect(page.getByRole("button", { name: "Not for me", exact: true })).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "It’s sacred to me", exact: true }).click();
  await expect(page.getByRole("button", { name: "Unsure", exact: true })).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByRole("heading", { name: "Here’s what you told us." })).toBeVisible();
  await expect(page.getByTestId("card-marital_status")).toContainText("Married");
  await expect(page.getByTestId("card-hold_partnership")).toContainText("I want it, I feel I ought to. “It matters to my family.”");
  await expect(page.getByTestId("card-hold_children")).toContainText("It’s sacred to me");
  for (const id of ["name", "date_of_birth", "occupation", "closest_friend", "seeing_someone", "children"]) {
    await expect(page.getByTestId(`card-${id}`)).toBeVisible();
  }
  await expect(page.getByTestId("card-closest_friend")).toContainText("Sam");
  await shot(page, "6-review");
  await seriousAxe(page);
  const before = requests.length;
  await page.getByRole("button", { name: "Keep these" }).click();
  await expect(page.getByTestId("empty")).toHaveText("Nothing here yet, Ada. Whatever you write stays private to you.");
  await shot(page, "7-journal");
  for (const secret of ["Ada", "Sam", "Engineer"]) {
    expect(requests.slice(0, before).some((r) => r.includes(secret))).toBe(false);
  }
  await page.goto("/welcome");
  await expect(page).toHaveURL(/\/journal$/);
});

test("a signed-out visitor is sent to sign in", async ({ page }) => {
  await page.goto("/welcome");
  await expect(page).toHaveURL(/\/sign-in$/);
  await shot(page, "0-sign-in");
});

test("skipping everything keeps nothing", async ({ page }) => {
  await signUp(page);
  await skipRest(page);
  await expect(page.getByTestId("nothing")).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByTestId("empty")).toHaveText("Nothing here yet. Whatever you write stays private to you.");
});

test("edit and remove on the review", async ({ page }) => {
  await signUp(page);
  await answer(page, "What should we call you?", "Ada");
  await answer(page, "When were you born?", "1990-04-01");
  await answer(page, "What do you do?", "Engineer");
  await answer(page, "Who is your closest friend?", "Sam");
  await skipRest(page);
  await page.getByTestId("card-occupation").getByRole("button", { name: "Edit" }).click();
  await expect(page.getByLabel("What do you do?")).toHaveValue("Engineer");
  await page.getByLabel("What do you do?").fill("Teacher");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByTestId("card-occupation")).toContainText("Teacher");
  await expect(page.getByTestId("card-closest_friend")).toContainText("Sam");
  await page.getByRole("button", { name: "Remove Closest friend" }).click();
  await expect(page.getByTestId("card-closest_friend")).toHaveCount(0);
  await page.getByRole("button", { name: "Keep these" }).click();
  await expect(page.getByTestId("empty")).toHaveText("Nothing here yet, Ada. Whatever you write stays private to you.");
});

test("a future date of birth is refused in place", async ({ page }) => {
  await signUp(page);
  await answer(page, "What should we call you?", "Ada");
  await page.getByLabel("When were you born?").fill("2099-01-01");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.locator("#answer-date_of_birth-error")).toHaveText("A date of birth can't be in the future.");
  await expect(page.getByLabel("When were you born?")).toBeVisible();
  await shot(page, "error");
});

test("the whole flow works from the keyboard", async ({ page }) => {
  await signUp(page);
  await page.keyboard.type("Ada");
  await page.keyboard.press("Enter");
  await page.keyboard.type("1990-04-01".replaceAll("-", ""));
  await page.keyboard.press("Enter");
  await page.keyboard.type("Engineer");
  await page.keyboard.press("Enter");
  await page.keyboard.type("Sam");
  await page.keyboard.press("Enter");
  const pickFirstThenNext = async () => {
    await page.keyboard.press("Enter");
    for (let i = 0; i < 20; i++) {
      if ((await page.evaluate(() => document.activeElement?.textContent)) === "Next") break;
      await page.keyboard.press("Tab");
    }
    await page.keyboard.press("Enter");
  };
  await pickFirstThenNext();
  await pickFirstThenNext();
  await page.keyboard.type("0");
  await page.keyboard.press("Enter");
  await pickFirstThenNext();
  await pickFirstThenNext();
  await expect(page.getByTestId("card-marital_status")).toContainText("Single");
  await expect(page.getByTestId("card-hold_children")).toContainText("I want it");
  await expect(page.getByTestId("card-closest_friend")).toContainText("Sam");
});
