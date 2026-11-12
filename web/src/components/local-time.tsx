"use client";

import { useEffect, useState } from "react";

/** A moment in the person's own time zone, drawn after mounting (the server does not know the zone). */
export function LocalTime({ iso, only }: { iso: string; only?: "time" }) {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => setText(only === "time" ? new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })), [iso, only]);
  return <time dateTime={iso} suppressHydrationWarning>{text ?? ""}</time>;
}
