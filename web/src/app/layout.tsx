import "./globals.css";
import type { ReactNode } from "react";
import { Inter } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

export const metadata = { title: "Reflect", description: "Your life, in your own records." };

/** Every page in shadcn's look (src/app/globals.css). */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={sans.variable}>
      <body className="min-h-screen">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
