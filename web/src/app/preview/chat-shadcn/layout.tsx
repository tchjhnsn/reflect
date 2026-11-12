import type { ReactNode } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import "../shadcn.css";

/** A preview of the chat in shadcn's default look with AI Elements, beside the app's own look at /chat. */
export default function ShadcnPreviewLayout({ children }: { children: ReactNode }) {
  return (
    <div className="shadcn min-h-screen">
      <TooltipProvider>{children}</TooltipProvider>
    </div>
  );
}
