import { describe, expect, it } from "vitest";
import { gaps, numberedTranscript, parseQuiz, quizPrompt } from "./quiz";

const transcript = [
  { at: 0, heading: true, who: null, text: "Chapter 1. Introduction" },
  { at: null, heading: false, who: "Professor Kagan", text: "Death is bad for the one who dies  because it deprives them." },
  { at: null, heading: false, who: null, text: "Epicurus disagreed." },
];

describe("quizzing yourself on a lecture", () => {
  it("numbers the transcript and cuts it at a length", () => {
    expect(numberedTranscript(transcript)).toBe("[0] ## Chapter 1. Introduction\n[1] Professor Kagan: Death is bad for the one who dies because it deprives them.\n[2] Epicurus disagreed.");
    expect(numberedTranscript(transcript, 40)).toBe("[0] ## Chapter 1. Introduction");
    expect(quizPrompt("Death", transcript)).toContain('"Death"');
  });
  it("reads the questions from a reply and leaves out what is malformed", () => {
    const reply = 'Here:\n```json\n{"questions":[{"kind":"cause","question":"Why is death bad?","answer":"It deprives.","passage":1},{"kind":"odd","question":"Who disagreed?","answer":"Epicurus.","passage":"2"},{"question":"","answer":"x"},{"question":"Out of range?","answer":"Yes.","passage":9}]}\n```';
    expect(parseQuiz(reply, 3)).toEqual([
      { kind: "cause", question: "Why is death bad?", answer: "It deprives.", passage: 1 },
      { kind: "recall", question: "Who disagreed?", answer: "Epicurus.", passage: 2 },
      { kind: "recall", question: "Out of range?", answer: "Yes.", passage: null },
    ]);
    expect(parseQuiz("no json", 3)).toEqual([]);
    expect(parseQuiz("{oops}", 3)).toEqual([]);
  });
  it("lists what the person did not know yet", () => {
    const qs = parseQuiz('{"questions":[{"question":"A?","answer":"a"},{"question":"B?","answer":"b"},{"question":"C?","answer":"c"}]}', 0);
    expect(gaps(qs, [{ seq: 0, answer: "a", knew: true }, { seq: 2, answer: "?", knew: false }]).map((g) => g.question)).toEqual(["C?"]);
  });
});
