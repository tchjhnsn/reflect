// The fixed onboarding wording. Deterministic: no model writes or picks these lines.
export const COPY: Record<string, { prompt: string; placeholder: string; note?: string; long?: boolean; date?: boolean }> = {
  name: { prompt: "What should we call you?", placeholder: "Your name" },
  date_of_birth: { prompt: "When were you born?", placeholder: "YYYY-MM-DD", date: true },
  occupation: { prompt: "What do you do?", placeholder: "Your work, in your words" },
  closest_friend: {
    prompt: "Who is your closest friend?",
    placeholder: "Their name",
    note: "Only you see this. Write only what you want to.",
  },
  puzzle: { prompt: "What is the puzzle of your life right now?", placeholder: "Tell me more.", long: true },
};

export const LABELS: Record<string, string> = {
  name: "Name",
  date_of_birth: "Born",
  occupation: "Work",
  closest_friend: "Closest friend",
  puzzle: "Puzzle",
};
