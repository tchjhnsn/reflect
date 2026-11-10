import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { expect, test, type Page } from "@playwright/test";
import { onboardAs, settle } from "./helpers";

const shots = process.env.SHOTS_DIR;
const axeSource = readFileSync(createRequire(import.meta.url).resolve("axe-core/axe.min.js"), "utf8");

async function shot(page: Page, name: string) {
  await settle(page);
  if (shots) await page.screenshot({ path: `${shots}/${test.info().project.name}-journal-${name}.png`, fullPage: true });
}

async function postShort(page: Page, text: string) {
  await page.getByLabel("What would you like to remember?").fill(text);
  await page.getByRole("button", { name: "Post" }).click();
  await expect(page.getByTestId("entry").first()).toContainText(text);
}

test("post, tell me more, and close an entry; it all survives a reload", async ({ page }) => {
  await onboardAs(page, "Ada");
  await shot(page, "1-empty");
  await postShort(page, "Rain at the park. We went anyway.");
  await expect(page.getByLabel("What would you like to remember?")).toHaveValue("");
  const entry = page.getByTestId("entry").first();
  await entry.getByRole("button", { name: "Tell me more." }).click();
  await entry.getByLabel("Tell me more.").fill("Sam brought a towel for the ball.");
  await shot(page, "2-more");
  await entry.getByRole("button", { name: "Add" }).click();
  await expect(entry).toContainText("Sam brought a towel for the ball.");
  await entry.getByRole("button", { name: "I’m done" }).click();
  await expect(entry.getByText("Done", { exact: true })).toBeVisible();
  await expect(entry.getByRole("button", { name: "Tell me more." })).toHaveCount(0);
  await page.reload();
  await expect(page.getByTestId("entry").first()).toContainText("Rain at the park. We went anyway.");
  await expect(page.getByTestId("entry").first()).toContainText("Sam brought a towel for the ball.");
  await expect(page.getByRole("heading", { name: "Today" })).toBeVisible();
  await shot(page, "3-timeline");
});

test("keep writing opens a long page and posts it whole", async ({ page }) => {
  await onboardAs(page, "Ada");
  await page.getByRole("button", { name: "Keep writing" }).click();
  const long = "First line.\n\nA second paragraph, much longer, about the whole afternoon.";
  await page.getByLabel("What would you like to remember?").fill(long);
  await shot(page, "4-long");
  await page.getByRole("button", { name: "Post" }).click();
  await expect(page.getByTestId("entry").first()).toContainText("A second paragraph");
});

test("newest posts come first and blank posts are refused", async ({ page }) => {
  await onboardAs(page, "Ada");
  await expect(page.getByRole("button", { name: "Post" })).toBeDisabled();
  await postShort(page, "First.");
  await postShort(page, "Second.");
  await expect(page.getByTestId("entry").first()).toContainText("Second.");
  await expect(page.getByTestId("entry").nth(1)).toContainText("First.");
});

test("one person never sees another's journal", async ({ browser }) => {
  const a = await browser.newPage();
  await onboardAs(a, "Ada");
  await postShort(a, "Only Ada should see this.");
  const b = await browser.newPage();
  await onboardAs(b, "Kai");
  await expect(b.getByTestId("empty")).toBeVisible();
  await expect(b.getByText("Only Ada should see this.")).toHaveCount(0);
});

async function seriousAxe(page: Page) {
  await settle(page);
  await page.addScriptTag({ content: axeSource });
  const ids = await page.evaluate(async () => {
    const r = await (window as unknown as { axe: { run: () => Promise<{ violations: { id: string; impact: string }[] }> } }).axe.run();
    return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id);
  });
  expect(ids).toEqual([]);
}

test("the journal has no serious accessibility issues", async ({ page }) => {
  await onboardAs(page, "Ada");
  await postShort(page, "Checking.");
  await settle(page);
  await page.addScriptTag({ content: axeSource });
  const ids = await page.evaluate(async () => {
    const r = await (window as unknown as { axe: { run: () => Promise<{ violations: { id: string; impact: string }[] }> } }).axe.run();
    return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id);
  });
  expect(ids).toEqual([]);
});

test("reflect on a closed entry: pick a frame, answer or skip, and what you ought stays apart", async ({ page }) => {
  await onboardAs(page, "Ada");
  await postShort(page, "I said yes to the move.");
  const entry = page.getByTestId("entry").first();
  await entry.getByRole("button", { name: "I’m done" }).click();
  await entry.getByRole("button", { name: "Reflect on it" }).click();
  await expect(entry.getByText("I noticed Emotion and Attachment. Is that right? Change anything that isn’t.")).toBeVisible();
  await expect(entry.getByRole("button", { name: "Emotion", exact: true })).toHaveAttribute("aria-pressed", "true");
  await entry.getByRole("button", { name: "Attachment", exact: true }).click();
  await entry.getByRole("button", { name: "Embodiment", exact: true }).click();
  await shot(page, "3-labels");
  await seriousAxe(page);
  await entry.getByRole("button", { name: "Keep labels" }).click();
  await expect(entry.getByTestId("labels")).toHaveText("Emotion · Embodiment");
  await shot(page, "3-frames");
  await entry.getByRole("button", { name: /^A decision/ }).click();
  await expect(entry.getByText("A decision · 1 of 17")).toBeVisible();
  await entry.getByLabel("Was this something you chose").fill("I chose it.");
  await entry.getByRole("button", { name: "Next" }).click();
  for (let i = 0; i < 4; i++) await entry.getByRole("button", { name: "Skip" }).click();
  await entry.getByLabel("Did you feel you ought to? Why or why not?").fill("My mother expects it.");
  await shot(page, "4-question");
  await entry.getByRole("button", { name: "Next" }).click();
  await expect(entry.getByText("A decision · 7 of 17")).toBeVisible();
  await entry.getByRole("button", { name: "Stop here" }).click();
  const r = entry.getByTestId("reflection");
  await expect(r.getByTestId("reflection-is")).toContainText("I chose it.");
  await expect(r.getByTestId("reflection-is")).not.toContainText("mother");
  await expect(r.getByTestId("reflection-ought")).toContainText("My mother expects it.");
  await page.reload();
  await expect(page.getByTestId("reflection-ought")).toContainText("My mother expects it.");
  await shot(page, "5-reflection");
});

test("a self-term's basis is the person's to correct", async ({ page }) => {
  await onboardAs(page, "Ada");
  await postShort(page, "I keep telling myself I'm the independent one.");
  const entry = page.getByTestId("entry").first();
  await entry.getByRole("button", { name: "I’m done" }).click();
  await entry.getByRole("button", { name: "Reflect on it" }).click();
  const basis = entry.getByLabel("You called yourself “the independent one”. What is that comparison made through?");
  await expect(basis).toHaveValue("will");
  await basis.selectOption("imagination");
  await shot(page, "6-self-term");
  await entry.getByRole("button", { name: "Keep labels" }).click();
  await expect(entry.getByText("What would you like to look at?")).toBeVisible();
});
