import type { ReactNode } from "react";

/** The chat fills the screen; the theme comes from the root layout. */
export default function ChatLayout({ children }: { children: ReactNode }) {
  return <div className="min-h-screen">{children}</div>;
}
