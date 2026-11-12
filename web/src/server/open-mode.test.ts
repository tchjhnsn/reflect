import { describe, expect, it } from "vitest";
import { openMode } from "./open-mode";

describe("open mode", () => {
  it("is on for the local dev server and Vercel previews, off in production, and can be forced", () => {
    expect(openMode({ NODE_ENV: "development" })).toBe(true);
    expect(openMode({ NODE_ENV: "production", VERCEL: "1", VERCEL_ENV: "preview" })).toBe(true);
    expect(openMode({ NODE_ENV: "production", VERCEL: "1", VERCEL_ENV: "production" })).toBe(false);
    expect(openMode({ NODE_ENV: "production" })).toBe(false);
    expect(openMode({ NODE_ENV: "development", REFLECT_OPEN: "off" })).toBe(false);
    expect(openMode({ NODE_ENV: "production", REFLECT_OPEN: "on" })).toBe(true);
  });
});
