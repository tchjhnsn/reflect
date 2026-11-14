import { redirect } from "next/navigation";

/** The log moved into the journal as one of its lenses. */
export default function LogPage() {
  redirect("/journal/log");
}
