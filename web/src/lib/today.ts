"use client";

import { useEffect, useState } from "react";
import type { IsoDate } from "@reflect/core/person";

export function localToday(): IsoDate {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` as IsoDate;
}

/** The person's own date, known only after the page reaches their device. */
export function useToday(): IsoDate | null {
  const [today, setToday] = useState<IsoDate | null>(null);
  useEffect(() => setToday(localToday()), []);
  return today;
}

export function longDate(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
}

export function shortDate(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
