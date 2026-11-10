// Words for stored ids on the Me page. Deterministic: no model writes these lines.
/** Words for each stored choice id. */
export const CHOICE_LABELS: Record<string, string> = {
  single: "Single",
  dating: "Dating",
  partnered: "Partnered",
  married: "Married",
  separated: "Separated",
  divorced: "Divorced",
  widowed: "Widowed",
  rather_not_say: "Rather not say",
  yes: "Yes",
  no: "No",
  complicated: "It’s complicated",
};

export const HOLD_LABELS: Record<string, string> = {
  desire: "I want it",
  will: "I’m working toward it",
  duty: "I feel I ought to",
  sacred: "It’s sacred to me",
  not_for_me: "Not for me",
  unsure: "Unsure",
};

/** An answer as words, for the Me page. */
export function showAnswer(id: string, value: string): string {
  if (CHOICE_LABELS[value] && (id === "marital_status" || id === "seeing_someone")) return CHOICE_LABELS[value]!;
  if (id === "children") return value === "0" ? "None" : value;
  if (id.startsWith("hold_")) {
    const [picks = "", ...rest] = value.split(";");
    const words = rest.join(";").trim();
    const named = picks.split(",").map((x) => HOLD_LABELS[x.trim()] ?? x).join(", ");
    return words ? `${named}. “${words}”` : named;
  }
  return value;
}

