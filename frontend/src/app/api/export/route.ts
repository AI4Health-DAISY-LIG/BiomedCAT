import { toCsv } from "@/lib/csv";
import { selectRun, type RunResult } from "@/lib/runs";

/**
 * Downloads one run as JSON or CSV. Looked up by the filename the run recorded,
 * never by a path from the request, so no input reaches the filesystem.
 */
export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const file = params.get("file") ?? undefined;
  const format = params.get("format") ?? "json";

  const result = selectRun(file);
  if (!result) {
    return Response.json(
      { error: file ? `No run found for ${file}.` : "No runs available." },
      { status: 404 },
    );
  }

  const stem = result.run.file.replace(/\.[^.]+$/, "");

  if (format === "csv") {
    // The BOM is what makes Excel read the file as UTF-8 rather than Latin-1.
    return download("﻿" + entitiesCsv(result), "text/csv", `${stem}_entities.csv`);
  }

  return download(
    JSON.stringify(result, null, 2),
    "application/json",
    `${stem}_BiomedCAT.json`,
  );
}

function entitiesCsv(result: RunResult) {
  const { run, entities } = result;
  return toCsv([
    ["Deck", "Entity", "Type", "CURIE", "Segment"],
    ...entities.map((e) => [run.file, e.text, e.type, e.curie, e.segment]),
  ]);
}

function download(body: string, type: string, filename: string) {
  return new Response(body, {
    headers: {
      "Content-Type": `${type}; charset=utf-8`,
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
  });
}
