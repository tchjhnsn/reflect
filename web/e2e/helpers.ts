import { expect, type Page } from "@playwright/test";

export async function signUp(page: Page) {
  const email = `t${Date.now()}${Math.floor(Math.random() * 1e6)}@example.test`;
  await page.goto("/sign-in");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill("a long test password");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/journal$/);
  return email;
}

/** Sign up, then state a name on the Me page, the only place a name is set now. */
export async function signUpAs(page: Page, name: string) {
  await signUp(page);
  await page.goto("/me");
  await page.getByRole("button", { name: "Change Name" }).click();
  await page.getByLabel("Name", { exact: true }).fill(name);
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText(name);
  await page.goto("/journal");
}

export async function settle(page: Page) {
  await page.evaluate(() => Promise.all(document.getAnimations().map((a) => a.finished)));
}
