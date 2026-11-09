import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Reflect", description: "A private journal." };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
