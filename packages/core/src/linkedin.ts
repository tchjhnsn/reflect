// A LinkedIn data export ("Get a copy of your data" in LinkedIn's settings), read for what others say
// about what the person knows. LinkedIn's API gives other apps only a name and an email, so the export is
// the way in. It is a zip of CSV files; the ones read here are Endorsement_Received_Info.csv (who
// endorsed which skill), Recommendations_Received.csv (their words), Certifications.csv and Skills.csv
// (the person's own list, kept as subjects with no grasp). Columns are found by name, so a reordered or
// extended export still reads. The person sees what was found before anything is kept.

export type LinkedInExport = {
  endorsements: { on: string; skill: string; by: string; url?: string }[];
  recommendations: { by: string; role?: string; company?: string; on?: string; text: string }[];
  certifications: { name: string; issuer: string; url?: string; issued?: string; expires?: string }[];
  skills: string[];
};

/** RFC 4180 CSV: quoted fields, doubled quotes, newlines inside quotes. LinkedIn sometimes puts notes above the header. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  const t = text.replace(/^﻿/, "");
  for (let i = 0; i < t.length; i++) {
    const ch = t[i]!;
    if (quoted) {
      if (ch === '"' && t[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && t[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += ch;
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim()));
}

/** Rows as objects keyed by lowercased header, from the first row that has every wanted column. */
function table(text: string | undefined, wanted: string[]): Record<string, string>[] {
  if (!text) return [];
  const rows = parseCsv(text);
  const at = rows.findIndex((r) => wanted.every((w) => r.some((c) => c.trim().toLowerCase() === w)));
  if (at < 0) return [];
  const head = rows[at]!.map((c) => c.trim().toLowerCase());
  return rows.slice(at + 1).map((r) => Object.fromEntries(head.map((h, i) => [h, (r[i] ?? "").trim()])));
}

const MONTHS: Record<string, string> = { jan: "01", feb: "02", mar: "03", apr: "04", may: "05", jun: "06", jul: "07", aug: "08", sep: "09", oct: "10", nov: "11", dec: "12" };

/** LinkedIn dates come as `2024/03/01 10:00:00 UTC`, `03/01/24, 10:00 AM`, `Mar 2021` or `2021`. */
export function linkedInDate(s: string | undefined): string | undefined {
  const v = (s ?? "").trim();
  let m = /^(\d{4})[/-](\d{2})[/-](\d{2})/.exec(v);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  m = /^(\d{1,2})\/(\d{1,2})\/(\d{2,4})/.exec(v);
  if (m) return `${m[3]!.length === 2 ? `20${m[3]}` : m[3]}-${m[1]!.padStart(2, "0")}-${m[2]!.padStart(2, "0")}`;
  m = /^([A-Za-z]{3})[a-z]* (\d{4})$/.exec(v);
  if (m && MONTHS[m[1]!.toLowerCase()]) return `${m[2]}-${MONTHS[m[1]!.toLowerCase()]}`;
  m = /^(\d{4})$/.exec(v);
  return m ? m[1] : undefined;
}

const named = (files: Map<string, string>, name: string) => [...files].find(([k]) => k.split("/").pop()!.toLowerCase() === name.toLowerCase())?.[1];

export function readLinkedInExport(files: Map<string, string>): LinkedInExport {
  const endorsements = table(named(files, "Endorsement_Received_Info.csv"), ["skill name"])
    .filter((r) => !r["endorsement status"] || /accepted|visible|active/i.test(r["endorsement status"]!))
    .map((r) => ({ on: linkedInDate(r["endorsement date"]) ?? "", skill: r["skill name"]!, by: `${r["endorser first name"] ?? ""} ${r["endorser last name"] ?? ""}`.trim(), ...(r["endorser public url"] ? { url: r["endorser public url"] } : {}) }))
    .filter((e) => e.skill && e.by && e.on);
  const recommendations = table(named(files, "Recommendations_Received.csv"), ["text"])
    .filter((r) => !r.status || /visible|accepted/i.test(r.status))
    .map((r) => {
      const on = linkedInDate(r["creation date"]);
      return { by: `${r["first name"] ?? ""} ${r["last name"] ?? ""}`.trim(), ...(r["job title"] ? { role: r["job title"] } : {}), ...(r.company ? { company: r.company } : {}), ...(on ? { on } : {}), text: r.text! };
    })
    .filter((r) => r.by && r.text);
  const certifications = table(named(files, "Certifications.csv"), ["name"]).map((r) => {
    const issued = linkedInDate(r["started on"]);
    const expires = linkedInDate(r["finished on"]);
    return { name: r.name!, issuer: r.authority ?? "", ...(r.url ? { url: r.url } : {}), ...(issued ? { issued } : {}), ...(expires ? { expires } : {}) };
  }).filter((c) => c.name && c.issuer);
  const skills = table(named(files, "Skills.csv"), ["name"]).map((r) => r.name!).filter(Boolean);
  return { endorsements, recommendations, certifications, skills };
}
