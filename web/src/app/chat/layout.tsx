import type { ReactNode } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./shadcn.css";

/** The chat in shadcn's look with AI Elements (Torian chose it over the mono look, 2026-11-12 08:48Z). */
export default function ShadcnPreviewLayout({ children }: { children: ReactNode }) {
  return (
    <div className="shadcn min-h-screen">
      <TooltipProvider>{children}</TooltipProvider>
    </div>
  );
}
