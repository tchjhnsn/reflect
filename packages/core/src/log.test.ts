import { describe, expect, it } from "vitest";
import { actText, learnActs, logByDay } from "./log";

const facts = {
  books: [
    { title: "Summa Theologica", addedAt: "2026-11-10T09:00:00.000Z", startedOn: "2026-11-11", finishedOn: null },
    { title: "Meditations", addedAt: "2026-11-01T09:00:00.000Z", startedOn: "2026-11-02", finishedOn: "2026-11-11" },
  ],
  claims: [{ text: "Virtue is a mean", stances: [{ stance: "disbelief" as const, at: "2026-11-12T08:00:00.000Z" }, { stance: "belief" as const, at: "2026-11-11T08:00:00.000Z" }] }],
  grasps: [{ subject: "Stoicism", grasp: "explain" as const, at: "2026-11-11T10:00:00.000Z" }],
};

describe("the log", () => {
  it("reads acts from Learn's records: shelving is an intention, the first stance keeps a claim, later ones change it", () => {
    const acts = learnActs(facts);
    expect(acts.map(actText)).toEqual([
      "Put Summa Theologica on your shelves",
      "Started reading Summa Theologica",
      "Put Meditations on your shelves",
      "Started reading Meditations",
      "Finished Meditations",
      "Kept a claim as a belief: Virtue is a mean",
      "Changed your stance to a disbelief: Virtue is a mean",
      "Said you could explain the key ideas of Stoicism",
    ]);
    expect(acts.find((a) => a.kind === "intended")?.app).toBe("learn");
  });

  it("groups by day, newest first, timed acts before day-only ones", () => {
    const days = logByDay(learnActs(facts));
    expect(days.map((d) => d.day)).toEqual(["2026-11-12", "2026-11-11", "2026-11-10", "2026-11-02", "2026-11-01"]);
    expect(days[1]!.acts.map(actText)).toEqual([
      "Said you could explain the key ideas of Stoicism",
      "Kept a claim as a belief: Virtue is a mean",
      "Started reading Summa Theologica",
      "Finished Meditations",
    ]);
  });

  it("gives one act per kept status change, putting down included, and keeps a back-dated day", () => {
    const acts = learnActs({
      books: [{
        title: "Ulysses", addedAt: "2026-11-12T09:00:00.000Z", startedOn: "2026-11-12", finishedOn: null,
        statuses: [
          { status: "want", day: "2026-11-12", at: "2026-11-12T09:00:00.000Z" },
          { status: "reading", day: "2026-11-12", at: "2026-11-12T09:10:00.000Z" },
          { status: "abandoned", day: "2026-11-12", at: "2026-11-12T09:20:00.000Z" },
        ],
      }, {
        title: "Republic", addedAt: "2026-11-12T09:30:00.000Z", startedOn: "2019-03-01", finishedOn: "2019-03-01",
        statuses: [{ status: "read", day: "2019-03-01", at: "2026-11-12T09:30:00.000Z" }],
      }],
      claims: [],
      grasps: [],
    });
    expect(acts.map(actText)).toEqual(["Put Ulysses on your shelves", "Started reading Ulysses", "Put down Ulysses", "Finished Republic"]);
    expect(acts.at(-1)?.at).toBe("2019-03-01");
  });

  it("puts a timestamp on the person's own day", () => {
    const late = learnActs({ books: [], claims: [], grasps: [{ subject: "Logic", grasp: "gist", at: "2026-11-12T03:00:00.000Z" }] });
    expect(logByDay(late, -360)[0]!.day).toBe("2026-11-11");
  });
});

describe("items in the log", () => {
  it("shows adding, finishing, practising and lessons in the medium's words", async () => {
    const { learnActs, actText } = await import("./log");
    const acts = learnActs({ books: [], claims: [], grasps: [], items: [{ title: "Chess", done: "Played", addedAt: "2026-11-14T01:00:00.000Z", startedOn: null, finishedOn: "2026-11-14", sessions: ["2026-11-13"], lessons: ["2026-11-14T02:00:00.000Z"] }] });
    expect(acts.map(actText)).toEqual(["Added Chess to your shelves", "Played Chess", "Practised Chess", "Wrote what Chess taught you"]);
  });
});
