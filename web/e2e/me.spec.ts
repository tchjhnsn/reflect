import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { expect, test, type Page } from "@playwright/test";
import { PACKS } from "../src/core/packs";
import { onboardAs, settle } from "./helpers";

const shots = process.env.SHOTS_DIR;
const axeSource = readFileSync(createRequire(import.meta.url).resolve("axe-core/axe.min.js"), "utf8");

async function shot(page: Page, name: string) {
  await settle(page);
  if (shots) await page.screenshot({ path: `${shots}/${test.info().project.name}-me-${name}.png`, fullPage: true });
}

test("change a fact keeps the old value dated, and remove the friend", async ({ page }) => {
  await onboardAs(page, "Ada");
  await page.getByRole("link", { name: "Me" }).click();
  await expect(page).toHaveURL(/\/me$/);
  await expect(page.getByTestId("field-occupation")).toContainText("in your words");
  await page.getByRole("button", { name: "Change Work" }).click();
  await page.getByLabel("Work").fill("Teacher");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("field-occupation")).toContainText("Teacher");
  await page.getByRole("button", { name: "Change Work" }).click();
  await page.getByLabel("Work").fill("Nurse");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("field-occupation")).toContainText("Nurse");
  await page.getByRole("button", { name: "Change Work" }).click();
  await expect(page.getByRole("list", { name: "Earlier Work" })).toContainText("Teacher");
  await page.getByRole("button", { name: "Cancel" }).click();

  await page.getByRole("button", { name: "Change Born" }).click();
  await page.getByLabel("Born").fill("not a date");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Write the date as YYYY-MM-DD.")).toBeVisible();
  await page.getByLabel("Born").fill("1990-04-02");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("field-date_of_birth")).toContainText("1990");

  await page.getByRole("button", { name: "Change Closest friend" }).click();
  await page.getByLabel("Closest friend").fill("Sam");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("field-closest_friend")).toContainText("Sam");
  await page.getByRole("button", { name: "Change Closest friend" }).click();
  await page.getByRole("button", { name: "Remove" }).click();
  await expect(page.getByTestId("field-closest_friend")).toContainText("someone you can name");

  await page.getByRole("button", { name: "Change Name" }).click();
  await page.getByLabel("Name", { exact: true }).fill("Ada L.");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada L.");
  await page.reload();
  await expect(page.getByTestId("field-occupation")).toContainText("Nurse");
  await shot(page, "1-facts");
});

test("beliefs and values are stated, kept apart, and let go without disappearing", async ({ page }) => {
  await onboardAs(page, "Bo");
  await page.goto("/me");
  await page.getByLabel("Add belief").fill("People can change.");
  await page.getByRole("button", { name: "Add belief" }).click();
  await page.getByLabel("Add value").fill("Honesty matters.");
  await page.getByRole("button", { name: "Add value" }).click();
  await expect(page.getByTestId("belief")).toHaveText(/People can change\./);
  await expect(page.getByTestId("value")).toHaveText(/Honesty matters\./);
  await page.getByRole("button", { name: "I no longer hold: People can change." }).click();
  await expect(page.getByTestId("belief")).toHaveCount(0);
  await page.getByText("Held before", { exact: true }).click();
  await expect(page.getByText("People can change.")).toBeVisible();
  await shot(page, "2-beliefs");
});

test("prompts come only from chosen packs and are saved with the entry", async ({ page }) => {
  await onboardAs(page, "Cy");
  await expect(page.getByRole("button", { name: "Give me a prompt" })).toHaveCount(0);
  await page.goto("/me");
  const body = page.getByRole("button", { name: "Body check-in" });
  await body.click();
  await expect(body).toHaveAttribute("aria-pressed", "true");
  await page.waitForLoadState("networkidle");
  await page.reload();
  await expect(page.getByRole("button", { name: "Body check-in" })).toHaveAttribute("aria-pressed", "true");
  await shot(page, "3-packs");
  await page.getByRole("link", { name: "Journal" }).click();
  await page.getByRole("button", { name: "Give me a prompt" }).click();
  const bodyPack = PACKS.find((p) => p.id === "body")!.prompts.map((p) => p.text);
  const text = (await page.getByTestId("prompt").locator("label").textContent())!;
  expect(bodyPack).toContain(text);
  await page.getByRole("button", { name: "Another", exact: true }).click();
  const second = (await page.getByTestId("prompt").locator("label").textContent())!;
  expect(bodyPack).toContain(second);
  await page.getByLabel(second).fill("Tired, but my shoulders are loose.");
  await shot(page, "4-prompt");
  await page.getByRole("button", { name: "Post" }).click();
  const entry = page.getByTestId("entry").first();
  await expect(entry).toContainText(second);
  await expect(entry).toContainText("Tired, but my shoulders are loose.");
  await expect(page.getByTestId("prompt")).toHaveCount(0);
});

test("the Me page has no axe violations", async ({ page }) => {
  await onboardAs(page, "Di");
  await page.goto("/me");
  await settle(page);
  await page.addScriptTag({ content: axeSource });
  const result = await page.evaluate(async () => {
    const axe = (window as unknown as { axe: { run: () => Promise<{ violations: { id: string }[] }> } }).axe;
    return (await axe.run()).violations.map((v) => v.id);
  });
  expect(result).toEqual([]);
});
