// The fixed onboarding wording. Deterministic: no model writes or picks these lines.
export type Kind = "text" | "date" | "number" | "choice" | "hold";

export const COPY: Record<string, { prompt: string; placeholder: string; note?: string; date?: boolean; kind?: Kind }> = {
  name: { prompt: "What should we call you?", placeholder: "Your name" },
  date_of_birth: { prompt: "When were you born?", placeholder: "YYYY-MM-DD", date: true },
  occupation: { prompt: "What do you do?", placeholder: "Your work, in your words" },
  closest_friend: {
    prompt: "Who is your closest friend?",
    placeholder: "Their name",
    note: "Only you see this. Write only what you want to.",
  },
  marital_status: { prompt: "Are you married, or with someone?", placeholder: "", kind: "choice" },
  seeing_someone: { prompt: "Are you seeing anyone right now?", placeholder: "", kind: "choice" },
  children: { prompt: "Do you have children? How many?", placeholder: "0", kind: "number" },
  hold_partnership: {
    prompt: "How do you hold marriage or partnership?",
    placeholder: "In your own words, if you like",
    note: "Pick any that are true. Wanting, working toward, feeling you ought to, and holding sacred are different things.",
    kind: "hold",
  },
  hold_children: {
    prompt: "How do you hold having children?",
    placeholder: "In your own words, if you like",
    note: "Pick any that are true. Whether you have children or not.",
    kind: "hold",
  },
};

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

/** An answer as words, for the review and the Me page. */
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

export const LABELS: Record<string, string> = {
  name: "Name",
  date_of_birth: "Born",
  occupation: "Work",
  closest_friend: "Closest friend",
  marital_status: "Marital status",
  seeing_someone: "Seeing someone",
  children: "Children",
  hold_partnership: "Partnership",
  hold_children: "Children, held",
};
