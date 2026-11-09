import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://localhost:3000", trace: "off" },
  webServer: { command: "REFLECT_RATE_LIMIT=off pnpm start", url: "http://localhost:3000/welcome", reuseExistingServer: true },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "phone", use: { ...devices["iPhone 15"], browserName: "chromium" } },
  ],
});
