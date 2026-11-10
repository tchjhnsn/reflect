import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@reflect/core": fileURLToPath(new URL("../packages/core/src", import.meta.url)),
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: { fs: { allow: [".."] } },
  test: { include: ["src/**/*.test.ts", "../packages/core/src/**/*.test.ts"] },
});
