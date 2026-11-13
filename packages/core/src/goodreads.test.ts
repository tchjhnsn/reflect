import { describe, expect, it } from "vitest";
import { csvRows, readGoodreads } from "./goodreads.ts";

const EXPORT = `﻿Book Id,Title,Author,Author l-f,Additional Authors,ISBN,ISBN13,My Rating,Date Read,Date Added,Bookshelves,Exclusive Shelf
2767,"Nicomachean Ethics",Aristotle,"Aristotle, ",Roger Crisp,"=""0521630460""","=""9780521630467""",5,2024/03/09,2024/01/02,,read
3,"The City of God, Vol. 1",Augustine of Hippo,"Augustine",,"=""""","=""""",0,,2025/05/01,to-read,to-read
4,"A ""quoted"" title",Someone,"x",,"=""""","=""""",0,,2025/05/01,currently-reading,currently-reading
5,Mine,Me,"me",,"=""""","=""""",0,,2025/05/01,favorites,favorites
`;

describe("a Goodreads export", () => {
  it("reads quoted fields with commas, doubled quotes and newlines", () => {
    expect(csvRows('a,"b, c","d ""e""\nf"\r\n1,2,3\n')).toEqual([["a", "b, c", 'd "e"\nf'], ["1", "2", "3"]]);
  });

  it("turns each row into a book with its shelf as a status and the day it was read", () => {
    expect(readGoodreads(EXPORT)).toEqual([
      { title: "Nicomachean Ethics", authors: ["Aristotle", "Roger Crisp"], isbn: "9780521630467", status: "read", finishedOn: "2024-03-09", goodreadsId: "2767" },
      { title: "The City of God, Vol. 1", authors: ["Augustine of Hippo"], status: "want", goodreadsId: "3" },
      { title: 'A "quoted" title', authors: ["Someone"], status: "reading", goodreadsId: "4" },
      { title: "Mine", authors: ["Me"], status: "want", goodreadsId: "5" },
    ]);
  });

  it("refuses a file that is not an export", () => {
    expect(() => readGoodreads("name,age\nx,1\n")).toThrow(/not a Goodreads export/);
  });
});
