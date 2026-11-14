"use client";

import { useEffect, useState } from "react";
import { CopyIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

/** The invitation link, to send to the people you ask to examine you. */
export function CopyLink({ path }: { path: string }) {
  const [copied, setCopied] = useState(false);
  const [origin, setOrigin] = useState("");
  useEffect(() => setOrigin(window.location.origin), []);
  const url = `${origin}${path}`;
  return (
    <div className="flex max-w-xl gap-2">
      <Input readOnly value={url} aria-label="Invitation link" onFocus={(e) => e.currentTarget.select()} className="h-9" />
      <Button type="button" size="sm" variant="outline" className="h-9" onClick={() => { navigator.clipboard?.writeText(url).then(() => setCopied(true)).catch(() => {}); }}><CopyIcon />{copied ? "Copied" : "Copy"}</Button>
    </div>
  );
}
