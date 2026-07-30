/**
 * RFC 4180. Every field is quoted, not just the ones that need it: OCR segments
 * carry commas, quotes and newlines, and quoting everything never gets it wrong.
 */
export function toCsv(rows: (string | number | null)[][]) {
  return rows.map((row) => row.map(field).join(",")).join("\r\n");
}

function field(value: string | number | null) {
  const text = value === null ? "" : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}
