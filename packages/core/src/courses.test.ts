import { describe, expect, it } from "vitest";
import { courseInfo, isTextsHeading, textCitation, courseLecture, courseLectures, courseSite, linesOf, ocwCourse, ocwLectures, ocwSyllabus, oycCourse, oycLecture, oycLectures, passageAt, passagesFromCaptions, plain } from "./courses";

const OYC_COURSE = `<td class="views-field views-field-field-session-display-title"><a href="/philosophy/phil-176/lecture-2" hreflang="en">The Nature of Persons: Dualism vs. Physicalism</a></td>
<td class="views-field views-field-field-session-display-title"><a href="/philosophy/phil-176/lecture-1" hreflang="en">Course Introduction</a></td>`;
const OYC_LECTURE = `<h2 class="session-full__title"> Lecture 1 - Course Introduction </h2>
<h3 class="session-full__body-label"> Overview </h3> <p>Professor Kagan introduces the course.</p>
<iframe src="https://www.youtube.com/embed/p2J7wSuFRl8?si=x"></iframe>
<div id="inline_content"><h1 id="transcript-top">Death</h1><p><h3>Chapter 1. What This Class Is NOT [00:00:00]</h3><p><strong>Professor Shelly Kagan:</strong> All right, so this is Philosophy 176.</p><p>Now, as I say &amp; so on.</p><h3>Chapter 2. Common Views [00:06:38]</h3><p>More.</p></div></td>
<a href="/sites/default/files/courses/spring07/phil176/mp3/phil176_01_011607.mp3" rel="download">mp3</a>`;
const OCW_GALLERY = `<a class="video-link" href="/courses/18-06-linear-algebra-spring-2010/resources/lecture-1-the-geometry-of-linear-equations/"><div><img class="thumbnail" src="https://img.youtube.com/vi/J7DzL2_Na80/default.jpg" /></div><h5 class="video-title">Lecture 1: The geometry of linear equations</h5></a>
<a class="video-link" href="/courses/18-06-linear-algebra-spring-2010/resources/lecture-2-elimination-with-matrices/"><img src="https://img.youtube.com/vi/QVKj3LADCnA/default.jpg" /><h5 class="video-title">Lecture 2: Elimination with matrices</h5></a>`;
const OCW_LECTURE = `<iframe src="https://www.youtube.com/embed/J7DzL2_Na80"></iframe><a href="/courses/18-06-linear-algebra-spring-2010/d292_J7DzL2_Na80.vtt">captions</a> http://www.archive.org/download/MIT18.06S05_MP4/01.mp4`;
const VTT = "WEBVTT\n\n00:00:08.000 --> 00:00:12.000\nHi. This is the first lecture.\n\n00:00:20.000 --> 00:00:24.000\nAnd I'm Gilbert Strang.\n\n00:00:45.000 --> 00:00:50.000\nThe fundamental problem.\n";

describe("courses taken in Reflect", () => {
  it("knows which course pages Reflect can play", () => {
    expect(courseSite("https://oyc.yale.edu/philosophy/phil-176/lecture-3")).toEqual({ site: "oyc", url: "https://oyc.yale.edu/philosophy/phil-176" });
    expect(courseSite("https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/syllabus/")).toEqual({ site: "ocw", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/" });
    expect(courseSite("https://www.coursera.org/learn/x")).toBeNull();
    expect(courseSite("javascript:alert(1)")).toBeNull();
    expect(plain("<b>A</b> &amp; &#8217;B&#x2019;")).toBe("A & ’B’");
  });

  it("reads Open Yale lectures in order, with video, audio and the transcript in chapters", () => {
    expect(oycLectures(OYC_COURSE, "https://oyc.yale.edu/philosophy/phil-176").map((l) => [l.n, l.title])).toEqual([[1, "Course Introduction"], [2, "The Nature of Persons: Dualism vs. Physicalism"]]);
    const l = oycLecture(OYC_LECTURE, "https://oyc.yale.edu/philosophy/phil-176/lecture-1");
    expect(l).toMatchObject({ title: "Lecture 1 - Course Introduction", overview: "Professor Kagan introduces the course.", youtube: "p2J7wSuFRl8", audio: "https://oyc.yale.edu/sites/default/files/courses/spring07/phil176/mp3/phil176_01_011607.mp3" });
    expect(l.transcript).toEqual([
      { at: 0, heading: true, who: null, text: "Chapter 1. What This Class Is NOT" },
      { at: null, heading: false, who: "Professor Shelly Kagan", text: "All right, so this is Philosophy 176." },
      { at: null, heading: false, who: null, text: "Now, as I say & so on." },
      { at: 398, heading: true, who: null, text: "Chapter 2. Common Views" },
      { at: null, heading: false, who: null, text: "More." },
    ]);
    expect(passageAt(l.transcript, 10)).toBe(0);
    expect(passageAt(l.transcript, 400)).toBe(3);
  });

  it("reads MIT OpenCourseWare galleries and lectures, with captions as passages", async () => {
    const refs = ocwLectures(OCW_GALLERY, "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/");
    expect(refs.map((r) => [r.n, r.youtube])).toEqual([[1, "J7DzL2_Na80"], [2, "QVKj3LADCnA"]]);
    expect(passagesFromCaptions(VTT)).toEqual([
      { at: 8, heading: false, who: null, text: "Hi. This is the first lecture. And I'm Gilbert Strang." },
      { at: 45, heading: false, who: null, text: "The fundamental problem." },
    ]);
    const pages: Record<string, string> = {
      "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/": `<a href="/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/">Videos</a>`,
      "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/": OCW_GALLERY,
      [refs[0]!.url]: OCW_LECTURE,
      "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/d292_J7DzL2_Na80.vtt": VTT,
    };
    const f = async (u: string) => pages[u] ?? null;
    const c = await courseLectures("https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", f);
    expect(c?.lectures).toHaveLength(2);
    const l = await courseLecture("ocw", c!.lectures[0]!, f);
    expect(l).toMatchObject({ title: "Lecture 1: The geometry of linear equations", youtube: "J7DzL2_Na80", video: "https://www.archive.org/download/MIT18.06S05_MP4/01.mp4" });
    expect(l?.transcript).toHaveLength(2);
    expect(await courseLectures("https://example.com/x", f)).toBeNull();
  });

  it("reads a course's description and syllabus, to explore it before taking it", async () => {
    expect(linesOf("<p>One.</p><p>Two<br/>Three</p><ul><li>Four</li></ul>")).toEqual(["One.", "Two", "Three", "Four"]);
    const yale = `<title>Death | Open Yale Courses</title>
<div class="views-field"><span class="views-label views-label-field-about-the-course">About the Course</span> <div class="field-content"><p>There is one way to die.</p><p>Second paragraph.</p></div></div>
<div class="views-field"><span class="views-label views-label-field-professor">Professor</span> <div class="field-content">Shelly Kagan</div></div>
<div class="views-field"><span class="views-label views-label-field-texts">Texts</span> <div class="field-content"><p>Plato, Phaedo</p><p>Tolstoy, The Death of Ivan Ilych</p><p>For more, click here.</p></div></div>
<div class="views-field"><span class="views-label views-label-field-grading">Grading</span> <div class="field-content"><p>Papers: 50%</p></div></div>`;
    expect(oycCourse(yale)).toEqual({
      title: "Death", by: "Shelly Kagan", image: null, about: ["There is one way to die.", "Second paragraph."],
      syllabus: [{ heading: "Texts", lines: ["Plato, Phaedo", "Tolstoy, The Death of Ivan Ilych"] }, { heading: "Grading", lines: ["Papers: 50%"] }],
    });
    const home = `<meta property="og:image" content="https://ocw.mit.edu/x.jpg" /><h1> <a href="/courses/18-06/">Linear Algebra</a></h1><div id="full-description"><p>A basic subject on matrix theory.</p></div>`;
    expect(ocwCourse(home)).toEqual({ title: "Linear Algebra", by: null, image: "https://ocw.mit.edu/x.jpg", about: ["A basic subject on matrix theory."] });
    const syllabus = `<main id="course-content-section"><h2>Course Meeting Times</h2><p>Lectures: 3 sessions / week</p><h2>Textbook</h2><p>Strang, Introduction to Linear Algebra.</p><h2>Empty</h2></main>`;
    expect(ocwSyllabus(syllabus)).toEqual([{ heading: "Course Meeting Times", lines: ["Lectures: 3 sessions / week"] }, { heading: "Textbook", lines: ["Strang, Introduction to Linear Algebra."] }]);
    const base = "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/";
    const pages: Record<string, string> = { [base]: home + `<a href="/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/">Videos</a>`, [`${base}video_galleries/video-lectures/`]: OCW_GALLERY, [`${base}pages/syllabus/`]: syllabus };
    const info = await courseInfo(base, async (u) => pages[u] ?? null);
    expect(info).toMatchObject({ site: "ocw", title: "Linear Algebra", syllabus: [{ heading: "Course Meeting Times" }, { heading: "Textbook" }] });
    expect(info?.lectures).toHaveLength(2);
    expect(await courseInfo("https://example.com/x", async () => null)).toBeNull();
  });

  it("reads a syllabus's texts as books to find", () => {
    expect(["Texts", "Required Texts:", "Text", "Readings", "Textbook"].every(isTextsHeading)).toBe(true);
    expect(isTextsHeading("Grading")).toBe(false);
    expect(textCitation("Plato, Phaedo")).toEqual({ title: "Phaedo", author: "Plato", isbn: null });
    expect(textCitation("Browning, Christopher. Ordinary Men. New York: Harper Perennial, 1998.")).toEqual({ title: "Ordinary Men", author: "Christopher Browning", isbn: null });
    expect(textCitation('Barnes, Julian. "The Dream." In History of the World in 10 ½ Chapters.')).toEqual({ title: "History of the World in 10 ½ Chapters", author: "Julian Barnes", isbn: null });
    expect(textCitation('Brandt, Richard. "Suicide." In Moral Problems. Edited by James Rachels. New York: Harper, 1979.')).toEqual({ title: "Moral Problems", author: null, isbn: null });
    expect(textCitation("The readings are assigned in: Strang, Gilbert. Introduction to Linear Algebra. 4th ed. Wellesley-Cambridge Press, 2009. ISBN: 9780980232714.")).toEqual({ title: "Introduction to Linear Algebra", author: "Gilbert Strang", isbn: "9780980232714" });
    expect(textCitation("Course Packet:")).toBeNull();
    expect(textCitation("NOTE: More material is in Professor Strang’s 2014 textbook.")).toBeNull();
  });
});
