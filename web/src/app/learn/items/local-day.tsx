"use client";

import { useEffect, useState } from "react";

/** A hidden field holding the person's own date, so a day of learning is kept in their time zone. */
export function LocalDay() {
  const [day, setDay] = useState("");
  useEffect(() => {
    const d = new Date();
    setDay(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`);
  }, []);
  return <input type="hidden" name="day" value={day} />;
}
