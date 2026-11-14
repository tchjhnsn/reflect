import { describe, expect, it } from "vitest";
import { browseWay, findMedium, isBrowseMedium, oycCatalog, playsHere } from "./browse-media";

const OYC = `<tr><td class="views-field views-field-title"><a href="/philosophy">Philosophy</a></td><td class="views-field views-field-field-course-number"><a href="/philosophy/phil-176">PHIL 176</a></td>
<td class="views-field views-field-title-1"><a href="/philosophy/phil-176">Death</a></td><td class="views-field views-field-field-course-professor-last-name">Shelly Kagan </td></tr>
<tr><td class="views-field views-field-title-1"><a href="/economics/econ-252-08">Financial Markets (2008)</a></td></tr>`;
const MIT = JSON.stringify({ results: [{ title: "Linear Algebra", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", image: { url: "https://ocw.mit.edu/x.jpg" }, runs: [{ instructors: [{ full_name: "Prof. Gilbert Strang" }] }] }, { title: "Elsewhere", url: "https://example.com/x" }] });

describe("Browse beyond books", () => {
  it("reads Yale's catalog and MIT's courses with videos, and says what plays in Reflect", async () => {
    expect(oycCatalog(OYC)).toEqual([
      { medium: "course", title: "Death (PHIL 176)", by: "Shelly Kagan", url: "https://oyc.yale.edu/philosophy/phil-176", source: "Open Yale Courses" },
      { medium: "course", title: "Financial Markets", url: "https://oyc.yale.edu/economics/econ-252-08", source: "Open Yale Courses" },
    ]);
    const f = async (u: string) => (u.startsWith("https://api.learn.mit.edu/") ? MIT : null);
    const all = (await findMedium("course", "", f, { oycHtml: OYC })).found;
    expect(all.map((c) => c.title)).toEqual(["Linear Algebra", "Death (PHIL 176)", "Financial Markets"]);
    expect(all[0]).toMatchObject({ by: "Gilbert Strang", image: "https://ocw.mit.edu/x.jpg" });
    expect((await findMedium("course", "death", f, { oycHtml: OYC })).found.map((c) => c.title)).toEqual(["Death (PHIL 176)", "Linear Algebra"]);
    expect(all.every(playsHere)).toBe(true);
    expect(playsHere({ medium: "lecture", url: "https://www.youtube.com/watch?v=kBdfcR-8hEY" })).toBe(true);
    expect(playsHere({ medium: "podcast", url: "https://podcasts.apple.com/us/podcast/x/id659155419" })).toBe(true);
    expect(playsHere({ medium: "film", url: "https://www.wikidata.org/wiki/Q1" })).toBe(false);
    expect(playsHere({ medium: "art", url: "https://www.artic.edu/artworks/28560", image: "https://www.artic.edu/iiif/2/x/full/400,/0/default.jpg" })).toBe(true);
  });

  it("places each medium in its way", () => {
    expect(browseWay("course")).toBe("teaching");
    expect(browseWay("debate")).toBe("dialectic");
    expect(browseWay("film")).toBe("imitation");
    expect(browseWay("art")).toBe("leisure");
    expect(isBrowseMedium("books")).toBe(true);
    expect(isBrowseMedium("tweet")).toBe(false);
  });
});
