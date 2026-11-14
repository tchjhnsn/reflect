// The model that writes a lecture's quiz from its transcript (the chat's model). It only writes questions with the
// lecture's own answers; it never judges the person.
import { generateText } from "ai";
import { parseQuiz, quizPrompt, type Question } from "@reflect/core/quiz";
import type { Passage } from "@reflect/core/courses";
import { chatModel } from "./chat-model";

export async function makeQuiz(title: string, transcript: readonly Passage[]): Promise<Question[]> {
  const model = chatModel();
  if (!model) throw new Error("Quizzes need the model, which is not set up here.");
  if (!transcript.length) throw new Error("This lecture has no transcript to make a quiz from.");
  for (let i = 0; i < 2; i++) {
    const { text } = await generateText({ model, prompt: quizPrompt(title, transcript) });
    const questions = parseQuiz(text, transcript.length);
    if (questions.length >= 3) return questions;
  }
  throw new Error("The quiz could not be written just now. Try again in a moment.");
}
