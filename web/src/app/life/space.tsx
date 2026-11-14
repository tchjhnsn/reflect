/** What every tile's workspace shares: its column and its section heads. */
export const SPACE = "grid grid-cols-[minmax(0,1fr)] gap-10";

export function Head({ title, note }: { title: string; note: string }) {
  return <div><h2 className="text-lg font-semibold tracking-tight">{title}</h2><p className="text-muted-foreground text-sm">{note}</p></div>;
}
