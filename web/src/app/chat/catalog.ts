import { defineCatalog } from "@json-render/core";
import { schema } from "@json-render/react/schema";
import { z } from "zod";

/**
 * What the chat may show: pieces of Reflect's own apps, and nothing else. Reflect builds every spec from
 * the app's records (spec.ts); json-render renders only these components, each prop checked by its schema.
 */
export const catalog = defineCatalog(schema, {
  components: {
    Stack: { props: z.object({}), description: "Pieces shown one under another" },
    Shelf: { props: z.object({ id: z.string(), name: z.string(), books: z.array(z.string()), subjects: z.array(z.string()) }), description: "One of the person's shelves, from Learn" },
    LibraryBook: { props: z.object({ title: z.string(), by: z.string(), shelf: z.string(), onShelves: z.boolean(), done: z.boolean().nullable(), failed: z.boolean().nullable() }), description: "A book from the open library, with Add to my shelves" },
    Claim: { props: z.object({ text: z.string(), stance: z.string(), holders: z.string() }), description: "A claim the person keeps, with their stance" },
    ProposeClaim: { props: z.object({ text: z.string(), stance: z.string(), holders: z.string(), subject: z.string().nullable(), done: z.boolean().nullable(), failed: z.boolean().nullable() }), description: "A claim to keep, which the person accepts with one tap" },
    Note: { props: z.object({ text: z.string() }), description: "A short line from Reflect, such as nothing found" },
  },
  actions: {
    add_book: { params: z.object({ key: z.string(), entry: z.string() }), description: "Put a library book on the person's shelves" },
    keep_claim: { params: z.object({ key: z.string(), claim: z.string() }), description: "Keep a claim with the person's stance" },
  },
});
