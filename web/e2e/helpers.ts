import { expect, type Page } from "@playwright/test";

export async function signUp(page: Page) {
  const email = `t${Date.now()}${Math.floor(Math.random() * 1e6)}@example.test`;
  await page.goto("/sign-in");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill("a long test password");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/welcome$/);
  return email;
}

export async function onboardAs(page: Page, name: string) {
  await signUp(page);
  await page.getByLabel("What should we call you?").fill(name);
  await page.getByRole("button", { name: "Next" }).click();
  while (!(await page.getByRole("button", { name: "Keep these" }).isVisible())) await page.getByRole("button", { name: "Skip" }).click();
  await page.getByRole("button", { name: "Keep these" }).click();
  await expect(page).toHaveURL(/\/journal$/);
}

export async function settle(page: Page) {
  await page.evaluate(() => Promise.all(document.getAnimations().map((a) => a.finished)));
}
