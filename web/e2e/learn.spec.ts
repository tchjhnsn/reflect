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

test("keep a credential and a LinkedIn export's vouches, shown first and kept apart from what you know", async ({ page }) => {
  await signUpAs(page, "Mo");
  await page.goto("/learn");
  await page.getByLabel("Name", { exact: true }).fill("PMP");
  await page.getByLabel("Issuer", { exact: true }).fill("PMI");
  await page.getByLabel("Subjects it covers", { exact: false }).fill("Project management");
  await page.getByRole("button", { name: "Keep the credential" }).click();
  await expect(page.getByText("PMP, PMI. Covers Project management")).toBeVisible();

  const { zipSync, strToU8 } = await import("fflate");
  const zip = zipSync({
    "Endorsement_Received_Info.csv": strToU8("Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,Endorser Public Url,Endorsement Status\n2024/03/01 10:00:00 UTC,Systems Thinking,Ana,Lee,,ACCEPTED\n"),
    "Recommendations_Received.csv": strToU8('First Name,Last Name,Company,Job Title,Text,Creation Date,Status\nAna,Lee,Acme,CTO,"Sees the whole system.",03/05/24,VISIBLE\n'),
  });
  await page.getByLabel("Your LinkedIn data export", { exact: false }).setInputFiles({ name: "export.zip", mimeType: "application/zip", buffer: Buffer.from(zip) });
  await page.getByRole("button", { name: "Read the export" }).click();
  await expect(page.getByRole("heading", { name: "Your LinkedIn export" })).toBeVisible();
  await expect(page.getByText("Systems Thinking: Ana Lee")).toBeVisible();
  await page.getByRole("button", { name: "Keep what is checked" }).click();
  await expect(page.getByRole("heading", { name: "Vouched for by others" })).toBeVisible();
  await expect(page.getByText("Systems Thinking, by 1")).toBeVisible();
  await expect(page.getByText("Sees the whole system.")).toBeVisible();
});
