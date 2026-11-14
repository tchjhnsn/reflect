"use client";

import type { ReactNode } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

/** A syllabus as tabs, one per part (structure, texts, requirements, grading), the row of tabs swiping on a phone. */
export function SyllabusTabs({ tabs }: { tabs: { id: string; label: string; content: ReactNode }[] }) {
  if (!tabs.length) return null;
  return (
    <Tabs defaultValue={tabs[0]!.id} className="gap-4">
      <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
        <TabsList className="w-max">
          {tabs.map((t) => <TabsTrigger key={t.id} value={t.id} className="px-3">{t.label}</TabsTrigger>)}
        </TabsList>
      </div>
      {tabs.map((t) => <TabsContent key={t.id} value={t.id}>{t.content}</TabsContent>)}
    </Tabs>
  );
}
