import { describe, expect, it } from "vitest";
import { parseClippings } from "./kindle.ts";

const file = `﻿Pride and Prejudice (Jane Austen)
- Your Highlight on page 1 | Location 7-8 | Added on Monday, January 1, 2024 10:00:00 AM

It is a truth universally acknowledged,
that a single man
==========
Pride and Prejudice (Jane Austen)
- Your Note on page 1 | Location 8 | Added on Monday, January 1, 2024 10:01:00 AM

The famous opening.
==========
Pride and Prejudice (Jane Austen)
- Your Bookmark on page 3 | Location 40 | Added on Monday, January 1, 2024 10:02:00 AM


==========
Meditations (Marcus Aurelius (trans. Hays))
- Your Highlight at location 120-121 | Added on Tuesday

You have power over your mind.
==========
No Author Book
- Your Highlight on page 2 | Location 9-9 | Added on Tuesday

Words.
==========
`;

describe("kindle clippings", () => {
  it("reads highlights with their book, place and note, and drops bookmarks", () => {
    expect(parseClippings(file)).toEqual([
      { title: "Pride and Prejudice", author: "Jane Austen", place: "page 1, location 7-8", text: "It is a truth universally acknowledged, that a single man", note: "The famous opening." },
      { title: "Meditations", author: "Marcus Aurelius (trans. Hays)", place: "location 120-121", text: "You have power over your mind." },
      { title: "No Author Book", author: "", place: "page 2, location 9-9", text: "Words." },
    ]);
  });

  it("keeps the last version of a highlight changed on the Kindle", () => {
    const twice = "A (B)\n- Your Highlight on page 1 | Location 1-2\n\nfirst\n==========\nA (B)\n- Your Highlight on page 1 | Location 1-2\n\nsecond\n==========\n";
    expect(parseClippings(twice).map((c) => c.text)).toEqual(["second"]);
  });
});
