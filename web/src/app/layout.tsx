import "./globals.css";
import type { ReactNode } from "react";
import { cookies } from "next/headers";
import { Fraunces, Inter, Newsreader } from "next/font/google";
import { THEME_COOKIE, asTheme } from "@/core/theme";

const serif = Newsreader({ subsets: ["latin"], style: ["normal", "italic"], variable: "--font-serif", display: "swap" });
const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const display = Fraunces({ subsets: ["latin"], weight: ["600"], variable: "--font-display", display: "swap" });

export const metadata = { title: "Reflect", description: "A private journal." };

export default async function RootLayout({ children }: { children: ReactNode }) {
  const theme = asTheme((await cookies()).get(THEME_COOKIE)?.value);
  return (
    <html lang="en" data-theme={theme} className={`${serif.variable} ${sans.variable} ${display.variable}`}>
      <body>{children}</body>
    </html>
  );
}
