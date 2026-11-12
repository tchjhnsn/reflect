import "./globals.css";
import type { ReactNode } from "react";
import { Inter, Newsreader } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
// The journal's writing face.
const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", display: "swap", style: ["normal", "italic"] });

export const metadata = { title: "Reflect", description: "Your life, in your own records." };

/** Every page in shadcn's look (src/app/globals.css). */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body className="min-h-screen">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
