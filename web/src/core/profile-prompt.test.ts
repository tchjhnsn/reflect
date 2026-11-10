import { describe, expect, it } from "vitest";
import { PROFILE_PROMPT, parseProfileReply } from "./profile-prompt";

const reply = (body: object, after = "") => "```json\n" + JSON.stringify(body) + "\n```\n" + after;
const base = { kind: "reflect.profile-source", version: 1, reported_by: { assistant: "Claude", basis: "saved memory", reported_at: "2026-11-10" } };
const said = { basis: "i-said", evidence: "We talked about it.", confidence: "high" };

describe("profile prompt", () => {
  it("asks the AI never to invent and to say where each thing comes from", () => {
    expect(PROFILE_PROMPT).toContain("Never invent.");
    expect(PROFILE_PROMPT).toContain('"i-said"');
    expect(PROFILE_PROMPT).toContain("Never estimate my age.");
    expect(PROFILE_PROMPT).toContain("reflect.profile-source");
  });

  it("reads a reply, keeping the note after the code block", () => {
    const r = parseProfileReply(
      reply(
        {
          ...base,
          facts: { name: { value: "Torian", ...said }, occupation: null, children: { value: 0, ...said } },
          biography: [{ section: "work", text: "Builds ventures.", ...said }],
          beliefs: [{ text: "Records decide.", ...said }],
          values: [{ text: "Craft", basis: "inferred", evidence: "", confidence: "low" }],
          puzzle: { statement: "What is a person for?", why: "It recurs.", confidence: "medium", alternatives: ["Or this"] },
          open_questions: ["Where were you born?"],
        },
        "I'm least sure of your values.",
      ),
    );
    if ("error" in r) throw new Error(r.error);
    expect(r.report.assistant).toBe("Claude");
    expect(r.report.facts.map((f) => [f.key, f.value])).toEqual([["name", "Torian"], ["children", "0"]]);
    expect(r.report.biography[0]).toMatchObject({ section: "work", when: null });
    expect(r.report.values[0]!.basis).toBe("inferred");
    expect(r.report.puzzle?.alternatives).toEqual(["Or this"]);
    expect(r.report.note).toBe("I'm least sure of your values.");
    expect(r.report.dropped).toBe(0);
  });

  it("drops entries without a source or outside the profile's values, and counts them", () => {
    const r = parseProfileReply(
      reply({
        ...base,
        facts: {
          marital_status: { value: "engaged", ...said },
          date_of_birth: { value: "about 1998", ...said },
          occupation: { value: "Founder", evidence: "x", confidence: "high" },
        },
        biography: [{ section: "dreams", text: "x", ...said }],
        beliefs: [{ text: "", ...said }],
        puzzle: { statement: "x" },
      }),
    );
    if ("error" in r) throw new Error(r.error);
    expect(r.report.facts).toEqual([]);
    expect(r.report.puzzle).toBeNull();
    expect(r.report.dropped).toBe(6);
  });

  it("refuses something that isn't a reply to this prompt", () => {
    expect(parseProfileReply("hello")).toHaveProperty("error");
    expect(parseProfileReply(reply({ kind: "naive.founder-puzzle-source" }))).toHaveProperty("error");
  });
});
