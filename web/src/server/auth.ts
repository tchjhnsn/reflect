import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { db } from "@/db/pool";

export const auth = betterAuth({
  database: db(),
  emailAndPassword: { enabled: true, minPasswordLength: 10 },
  // Browser tests sign up many accounts from one address; they turn the limit off explicitly.
  rateLimit: { enabled: process.env.REFLECT_RATE_LIMIT !== "off" },
  plugins: [nextCookies()],
});
