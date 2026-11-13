import { ListenProvider } from "./player";

/** Learn's pages share one audio player, so a book keeps playing from page to page. */
export default function LearnLayout({ children }: { children: React.ReactNode }) {
  return <ListenProvider>{children}</ListenProvider>;
}
