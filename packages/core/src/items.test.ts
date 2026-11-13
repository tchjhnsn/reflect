import { describe, expect, it } from "vitest";
import { allowedMoves, apply, eventText, fold, phaseOf, readMove, refuse, type ItemEvent } from "./items";

const at = (n: number) => `2026-11-12T10:0${n}:00Z`;
const stated: ItemEvent = { kind: "stated", text: "I read too little fiction.", from: { kind: "entry", id: "e1" }, at: at(0) };

describe("items", () => {
  it("goes Observe, Reflect, Examine, Plan, Act and keeps every event", () => {
    const item = fold([
      stated,
      { kind: "reflected", outcome: "held", at: at(1) },
      { kind: "examine", at: at(2) },
      { kind: "examine_step", step: "ground", text: "Fiction trains imagination for ethics.", at: at(3) },
      { kind: "examined", believed: "qualified", stands: "grounded", text: "I read too little fiction for what I want to write.", at: at(4) },
      { kind: "planned", text: "One novel a month.", at: at(5) },
      { kind: "scheduled", on: "2026-12-01", at: at(6) },
      { kind: "started", at: at(7) },
      { kind: "done", at: at(8) },
    ])!;
    expect(item.state).toBe("done");
    expect(item.statement).toBe("I read too little fiction for what I want to write.");
    expect(item.from).toEqual({ kind: "entry", id: "e1" });
    expect(item.notes).toHaveLength(1);
    expect(item.plan).toBe("One novel a month.");
    expect(item.events).toHaveLength(9);
    expect(phaseOf(item.state)).toBe("Act");
  });

  it("refuses moves the statechart does not allow, with the reason", () => {
    expect(refuse(null, { kind: "reflected", outcome: "held" })).toBe("Nothing is stated yet.");
    const item = fold([stated])!;
    expect(refuse(item, { kind: "planned", text: "x" })).toMatch(/observed cannot be planned/);
    expect(() => apply(item, { kind: "done", at: at(1) })).toThrow();
    expect(refuse(item, { kind: "reflected", outcome: "revised" })).toBe("Write how you hold it now.");
    expect(allowedMoves("observed")).toEqual(["reflected"]);
  });

  it("drops an item without erasing it", () => {
    const item = fold([stated, { kind: "reflected", outcome: "dropped", at: at(1) }])!;
    expect(item.state).toBe("dropped");
    expect(item.events).toHaveLength(2);
    expect(refuse(item, { kind: "examine" })).toMatch(/dropped/);
    expect(phaseOf(item.state)).toBeNull();
  });

  it("a withdrawn belief leaves the pipeline", () => {
    const item = fold([stated, { kind: "reflected", outcome: "unsure", at: at(1) }, { kind: "examine", at: at(2) }, { kind: "examined", believed: "withdrawn", stands: "unchanged", at: at(3) }])!;
    expect(item.state).toBe("dropped");
  });

  it("a missed act goes back to Observe only through observing why", () => {
    const acting = fold([stated, { kind: "reflected", outcome: "held", at: at(1) }, { kind: "planned", text: "Read nightly.", at: at(2) }, { kind: "scheduled", on: "2026-11-13", at: at(3) }, { kind: "started", at: at(4) }, { kind: "missed", text: "Phone in bed.", at: at(5) }])!;
    expect(acting.state).toBe("missed");
    expect(refuse(acting, { kind: "planned", text: "again" })).toMatch(/observe why/);
    expect(refuse(acting, { kind: "observed_why", text: " " })).toBe("Say why it was missed.");
    const again = apply(acting, { kind: "observed_why", text: "The phone wins at night.", at: at(6) });
    expect(again.state).toBe("observed");
    expect(again.statement).toBe("I read too little fiction.");
    expect(eventText(again.events.at(-1)!)).toBe("Observed why it was missed: The phone wins at night.");
  });

  it("examination is ongoing: a done item can be examined again", () => {
    const done = fold([stated, { kind: "reflected", outcome: "held", at: at(1) }, { kind: "planned", text: "p", at: at(2) }, { kind: "scheduled", on: "2026-11-13", at: at(3) }, { kind: "started", at: at(4) }, { kind: "done", at: at(5) }])!;
    expect(apply(done, { kind: "examine", at: at(6) }).state).toBe("examining");
  });

  it("reads a move from a form and rejects what is not a move", () => {
    expect(readMove({ kind: "reflected", outcome: "held" })).toEqual({ kind: "reflected", outcome: "held", text: undefined });
    expect(readMove({ kind: "reflected", outcome: "maybe" })).toBeNull();
    expect(readMove({ kind: "examine_step", step: "nope", text: "x" })).toBeNull();
    expect(readMove("x")).toBeNull();
  });
});
