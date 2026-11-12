import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

// The repository root, so the app can import the headless core in ../packages/core.
const root = fileURLToPath(new URL("..", import.meta.url));

const config: NextConfig = { reactStrictMode: true, agentRules: false, turbopack: { root }, outputFileTracingRoot: root };

export default config;
