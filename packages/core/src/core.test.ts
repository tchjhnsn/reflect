import { describe, expect, it } from "vitest";
import { HOLDS, ONBOARDING, parseOnboarding } from "./structure.ts";
import { EMPTY_PERSON, currentFact, currentRelationship, isoDate, stateRelationship, type IsoDate } from "./person.ts";
import { START, cleanHold, finish, reduce, type OnboardingState } from "./onboarding.ts";

const TODAY = "2026-11-09" as IsoDate;

function run(answers: string[]): OnboardingState {
  return answers.reduce((s, text) => reduce(s, { type: "answer", text, today: TODAY }), START);
}

describe("onboarding", () => {
  it("follows the order Lean exports", () => {
    expect(ONBOARDING.map((q) => q.id)).toEqual([
      "name", "date_of_birth", "occupation", "closest_friend",
      "marital_status", "seeing_someone", "children", "hold_partnership", "hold_children",
    ]);
  });

  it("files family facts in the profile and holds apart from it", () => {
    let s = START;
    for (const text of ["Ada", "", "", "", "married", "no", "02", "duty,desire; my family", "sacred"]) {
      s = reduce(s, { type: "answer", text, today: TODAY });
      expect(s.error).toBeNull();
    }
    const p = finish(s, EMPTY_PERSON, TODAY);
    expect(currentFact(p, "marital_status")).toBe("married");
    expect(currentFact(p, "children")).toBe("2");
    expect(p.holds.map((h) => [h.subject, h.value])).toEqual([["partnership", "duty,desire; my family"], ["children", "sacred"]]);
    expect(p.facts.some((f) => f.key.startsWith("hold"))).toBe(false);
    expect(HOLDS.find((h) => h.id === "duty")?.stance).toBe("normative");
  });

  it("refuses answers outside the choices", () => {
    const at = (n: number) => {
      let s = START;
      for (let i = 0; i < n; i++) s = reduce(s, { type: "skip" });
      return s;
    };
    expect(reduce(at(4), { type: "answer", text: "engaged", today: TODAY }).error?.message).toBe("Pick one.");
    expect(reduce(at(6), { type: "answer", text: "two", today: TODAY }).error?.message).toBe("Write a number.");
    expect(reduce(at(7), { type: "answer", text: "unsure,desire", today: TODAY }).error).not.toBeNull();
    expect(cleanHold("desire, will ;  ok ")).toEqual({ value: "desire,will; ok" });
  });

  it("files each answer where it belongs", () => {
    const p = finish(run(["Ada", "1990-04-01", "Engineer", "Sam"]), EMPTY_PERSON, TODAY);
    expect(currentFact(p, "name")).toBe("Ada");
    expect(currentFact(p, "date_of_birth")).toBe("1990-04-01");
    expect(currentFact(p, "occupation")).toBe("Engineer");
    expect(currentRelationship(p, "closest_friend")).toBe("Sam");
    expect([...p.facts, ...p.relationships].every((r) => r.source.kind === "stated")).toBe(true);
  });

  it("writes nothing when every question is skipped", () => {
    const s = run(["", "  ", "", ""]);
    expect(finish(s, EMPTY_PERSON, TODAY)).toEqual(EMPTY_PERSON);
  });

  it("writes only the friend when only the friend is answered", () => {
    const p = finish(run(["", "", "", "Sam"]), EMPTY_PERSON, TODAY);
    expect([p.facts.length, p.relationships.length]).toEqual([0, 1]);
  });

  it("keeps earlier answers when going back", () => {
    let s = run(["Ada", "1990-04-01", "Engineer"]);
    s = reduce(s, { type: "back" });
    s = reduce(s, { type: "answer", text: "Teacher", today: TODAY });
    expect(s.answers).toMatchObject({ name: "Ada", occupation: "Teacher" });
  });

  it("returns to the review after editing or removing one answer", () => {
    let s = run(["Ada", "1990-04-01", "Engineer", "Sam"]);
    s = reduce(s, { type: "edit", step: 2 });
    expect(s.answers.occupation).toBe("Engineer");
    s = reduce(s, { type: "answer", text: "Teacher", today: TODAY });
    expect(s.step).toBe(ONBOARDING.length);
    s = reduce(reduce(s, { type: "edit", step: 3 }), { type: "skip" });
    expect(s.step).toBe(ONBOARDING.length);
    expect(s.answers).toEqual({ name: "Ada", date_of_birth: "1990-04-01", occupation: "Teacher" });
  });

  it("refuses a future date of birth and stays on the question", () => {
    const s = run(["Ada", "2030-01-01"]);
    expect(s.step).toBe(1);
    expect(s.error?.question).toBe("date_of_birth");
  });

  it("closes the old closest friend when a new one is stated", () => {
    let p = stateRelationship(EMPTY_PERSON, "Sam", "closest_friend", "2026-01-01" as IsoDate);
    p = stateRelationship(p, "Kai", "closest_friend", TODAY);
    expect(p.relationships[0]?.validTo).toBe(TODAY);
    expect(currentRelationship(p, "closest_friend")).toBe("Kai");
  });

  it("refuses a malformed structure", () => {
    expect(() => parseOnboarding({ onboarding: [{ id: "name" }] })).toThrow(/onboarding\[0\]\.destination/);
    expect(() => parseOnboarding({})).toThrow(/onboarding/);
  });

  it("rejects impossible dates", () => {
    expect(isoDate("2026-02-30")).toBeNull();
    expect(isoDate("1990-04-01")).toBe("1990-04-01");
  });
});
