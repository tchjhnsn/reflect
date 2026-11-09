import { describe, expect, it } from "vitest";
import { ONBOARDING, parseOnboarding } from "./structure";
import { EMPTY_PERSON, currentFact, currentRelationship, isoDate, stateRelationship, type IsoDate } from "./person";
import { START, finish, reduce, type OnboardingState } from "./onboarding";

const TODAY = "2026-11-09" as IsoDate;

function run(answers: string[]): OnboardingState {
  return answers.reduce((s, text) => reduce(s, { type: "answer", text, today: TODAY }), START);
}

describe("onboarding", () => {
  it("follows the order Lean exports", () => {
    expect(ONBOARDING.map((q) => q.id)).toEqual(["name", "date_of_birth", "occupation", "closest_friend"]);
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
