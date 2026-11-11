import { describe, expect, it } from "vitest";
import { FRAMES, answer, current, frameById, isOpenQuestion, split, type ReflectState } from "./reflect.ts";

describe("reflecting on an entry", () => {
  it("every frame asks only open questions", () => {
    for (const f of FRAMES) {
      expect(f.questions.length).toBeGreaterThan(5);
      for (const q of f.questions) expect(isOpenQuestion(q.text)).toBe(true);
    }
  });

  it("the decision frame asks whether it was chosen first and how firmly it was held last, with one normative question", () => {
    const d = frameById("decision")!;
    expect(d.questions[0]?.facet).toBe("voluntary_involuntary");
    expect(d.questions.at(-1)?.facet).toBe("will");
    expect(d.questions.find((q) => q.facet === "conscience")?.stance).toBe("normative");
    expect(d.questions.filter((q) => q.stance === "normative")).toHaveLength(1);
  });

  it("refuses leading or prescriptive questions", () => {
    expect(isOpenQuestion("Weren't you angry?")).toBe(false);
    expect(isOpenQuestion("You should rest more?")).toBe(false);
    expect(isOpenQuestion("What did you feel")).toBe(false);
    expect(isOpenQuestion("What did you feel?")).toBe(true);
  });

  it("walks the questions, skips leave no answer, and stances stay apart", () => {
    let s: ReflectState = { frame: "feeling", step: 0, answers: [] };
    s = answer(s, "Relief.");
    s = answer(s, null);
    const n = frameById("feeling")!.questions.length;
    while (current(s)) s = answer(s, current(s)!.facet === "conscience" ? "No, it was fine to feel this." : " ");
    expect(s.step).toBe(n);
    expect(answer(s, "extra")).toBe(s);
    const { descriptive, normative } = split(s.answers);
    expect(descriptive.map((a) => a.text)).toEqual(["Relief."]);
    expect(normative.map((a) => a.facet)).toEqual(["conscience"]);
  });
});
