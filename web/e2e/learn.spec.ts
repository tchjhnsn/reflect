import { expect, test } from "@playwright/test";
import { signUpAs } from "./helpers";

test("keep a book, name what you remember of it, and rate your grasp; read alone is not known", async ({ page }) => {
  await signUpAs(page, "Lin");
  await page.goto("/learn");
  await expect(page.getByRole("heading", { name: "What you know" })).toBeVisible();
  await page.getByLabel("Title", { exact: true }).fill("Thinking in Systems");
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

test("read a memory sweep from your AI, keep what is right, and see inputs apart from your ideas", async ({ page }) => {
  await signUpAs(page, "Ada");
  await page.goto("/learn");
  const reply = {
    kind: "reflect.learn-source",
    version: 1,
    reported_by: { assistant: "Claude", reported_at: "2026-11-11" },
    books: [{ title: "Politics", authors: ["Aristotle"], engaged: "referenced", basis: "i-said", evidence: "you cite Book III", confidence: "high" }],
    films: [{ title: "Metropolis", year: 1927, kind: "film", engaged: "finished", basis: "you-inferred", evidence: "you described its city", confidence: "low" }],
    subjects: { shown: [{ name: "Political philosophy", how: "explained", evidence: "the six constitutions", confidence: "high" }], suggested: [{ name: "Urban studies", because: "cities", confidence: "medium" }] },
    ideas: [{ title: "Corrupt and correct cities", statement: "Aristotle's correct and corrupt forms, applied to urban development.", inputs: [{ title: "Politics", how: "the test of Book III" }], basis: "i-said", confidence: "high" }],
  };
  await page.getByLabel("Its reply").fill("Here you go.\n```json\n" + JSON.stringify(reply) + "\n```");
  await page.getByRole("button", { name: "Read the reply" }).click();
  await expect(page.getByRole("heading", { name: "From Claude's memory" })).toBeVisible();
  await expect(page.getByLabel(/film: Metropolis/)).not.toBeChecked();
  await page.getByRole("button", { name: "Keep the checked items" }).click();
  await expect(page.getByText("(kept)")).toHaveCount(4);

  await page.goto("/learn");
  await expect(page.getByRole("heading", { name: "Shown, not yet rated by you" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Suggested, not known" })).toBeVisible();
  await expect(page.getByRole("term").filter({ hasText: "Political philosophy" })).toHaveCount(0);
  await page.getByLabel("What it is").selectOption("talk");
  await page.getByLabel("Title, or for a conversation", { exact: false }).fill("Strong Towns");
  await page.getByLabel("Who:", { exact: false }).fill("Chuck Marohn");
  await page.getByRole("button", { name: "Keep it" }).click();
  await expect(page.getByText("(talk by Chuck Marohn)")).toBeVisible();

  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Ideas" }).click();
  await page.getByRole("link", { name: "Corrupt and correct cities" }).click();
  await expect(page.getByText("Aristotle's correct and corrupt forms, applied to urban development.").first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Politics" })).toBeVisible();
  const from = page.locator("form", { has: page.getByRole("button", { name: "Keep the input" }) });
  await from.getByLabel("Something you kept").selectOption({ label: "Strong Towns (kept)" });
  await from.getByLabel("maybe, I am not sure").check();
  await from.getByRole("button", { name: "Keep the input" }).click();
  await expect(page.getByText("Maybe from, not sure:")).toBeVisible();
  const rec = page.locator("form", { has: page.getByRole("button", { name: "Keep the recommendation" }) });
  await rec.getByLabel("Or in your own words").fill("The Death and Life of Great American Cities");
  await rec.getByLabel("Where it stands").selectOption("challenges");
  await rec.getByLabel("Who suggested it").fill("Claude");
  await rec.getByRole("button", { name: "Keep the recommendation" }).click();
  await expect(page.getByText("The Death and Life of Great American Cities")).toBeVisible();
  await expect(page.getByText(", suggested by Claude")).toBeVisible();
});
