// Tailwind runs only on files that import it: the shadcn preview (src/app/preview/shadcn.css). The new
// design's pages keep their plain CSS in globals.css.
export default { plugins: { "@tailwindcss/postcss": {} } };
