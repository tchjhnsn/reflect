import type { Metadata } from "next";
import { Onboarding } from "./onboarding";

export const metadata: Metadata = { title: "Welcome to Reflect" };

export default function WelcomePage() {
  return <Onboarding />;
}
