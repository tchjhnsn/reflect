import { createCipheriv, createDecipheriv, hkdfSync, randomBytes } from "node:crypto";

function key(userId: string): Buffer {
  const master = process.env.REFLECT_DATA_KEY;
  if (!master) throw new Error("REFLECT_DATA_KEY is not set");
  return Buffer.from(hkdfSync("sha256", Buffer.from(master, "base64"), Buffer.from(userId), "reflect/person/v1", 32));
}

/** AES-256-GCM with a key derived per person. Output is base64 of iv, tag and ciphertext. */
export function seal(userId: string, text: string): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key(userId), iv);
  const ct = Buffer.concat([cipher.update(text, "utf8"), cipher.final()]);
  return Buffer.concat([iv, cipher.getAuthTag(), ct]).toString("base64");
}

export function open(userId: string, sealed: string): string {
  const raw = Buffer.from(sealed, "base64");
  const decipher = createDecipheriv("aes-256-gcm", key(userId), raw.subarray(0, 12));
  decipher.setAuthTag(raw.subarray(12, 28));
  return Buffer.concat([decipher.update(raw.subarray(28)), decipher.final()]).toString("utf8");
}
