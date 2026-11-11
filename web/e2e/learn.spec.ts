import { expect, test } from "@playwright/test";
import { signUpAs } from "./helpers";

test("keep a book, name what you remember of it, and rate your grasp; read alone is not known", async ({ page }) => {
  await signUpAs(page, "Lin");
  await page.goto("/learn");
  await expect(page.getByRole("heading", { name: "What you know" })).toBeVisible();
  await page.getByLabel("Title").fill("Thinking in Systems");
  await page.getByLabel("Its key subjects", { exact: false }).fill("Systems thinking, Feedback loops");
  await page.getByRole("button", { name: "Keep the book" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Thinking in Systems" })).toBeVisible();

  await page.getByLabel("Without looking", { exact: false }).fill("Stocks and flows, and systems thinking.");
  await page.getByRole("button", { name: "Keep what I named" }).click();
  await expect(page.getByText("(you named it)")).toBeVisible();
  await expect(page.getByText("(you did not name it)")).toBeVisible();

  await page.getByRole("group", { name: /Systems thinking/ }).getByLabel("could explain the key ideas").check();
  await page.getByRole("button", { name: "Keep my ratings" }).click();
  await expect(page.getByText("last: could explain the key ideas")).toBeVisible();

  await page.getByRole("link", { name: "What you know" }).click();
  await expect(page.getByRole("term").filter({ hasText: "Systems thinking" })).toBeVisible();
  await expect(page.getByText("Not yet rated: Feedback loops")).toBeVisible();
});
