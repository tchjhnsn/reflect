import { betterAuth } from "better-auth";
import { jwt } from "better-auth/plugins";
import { nextCookies } from "better-auth/next-js";
import { mcp } from "@better-auth/mcp";
import { db } from "@/db/pool";
import { mcpResource, siteOrigins, siteUrl } from "./site";

/** What a person's own AI may ask for. "journal" covers reading entries and keeping readings. */
export const MCP_SCOPES = ["openid", "profile", "offline_access", "journal"] as const;

export const auth = betterAuth({
  database: db(),
  baseURL: siteUrl(),
  trustedOrigins: siteOrigins(),
  emailAndPassword: { enabled: true, minPasswordLength: 10 },
  // Browser tests sign up many accounts from one address; they turn the limit off explicitly.
  rateLimit: { enabled: process.env.REFLECT_RATE_LIMIT !== "off" },
  plugins: [
    jwt(),
    // Reflect as an OAuth server for MCP clients (Claude, ChatGPT and others): the person signs in to
    // Reflect and agrees on /consent, and the client gets a token bound to /api/mcp. Clients register
    // themselves (RFC 7591), since a person's AI app is not known in advance.
    mcp({
      loginPage: "/sign-in",
      consentPage: "/consent",
      resource: mcpResource(),
      scopes: [...MCP_SCOPES],
      clientRegistrationDefaultScopes: [...MCP_SCOPES],
      allowDynamicClientRegistration: true,
      allowUnauthenticatedClientRegistration: true,
    }),
    nextCookies(),
  ],
});
