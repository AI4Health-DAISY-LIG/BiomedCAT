import fs from "node:fs";
import path from "node:path";
import { OUTPUT_DIR, readDir } from "./paths";

// Shapes of Output/<deck>_BiomedCAT.json (schema_version 1.0).

export type Entity = {
  text: string;
  type: string; // one of the seven ENTITY_TYPES
  segment: string;
  curie: string | null; // null when Norm found no acceptable candidate
};

export type Slide = {
  page: number;
  text: string;
};

export type RunResult = {
  schema_version: string;
  run: {
    file: string;
    timestamp: string;
    elapsed_s: number;
    stages?: { ocr: number; ner: number; norm: number }; // seconds, absent on older runs
    models: { ocr: string; ner: string; norm: string };
    resolvers: { renci: string; arax: string; api_limit: number };
  };
  slides: Slide[];
  entities: Entity[];
};

/** Every run in the output directory, newest first. */
export function listRuns(): RunResult[] {
  return readDir(OUTPUT_DIR)
    .filter((f) => f.endsWith(".json"))
    .map((f) => read(path.join(OUTPUT_DIR, f)))
    .sort((a, b) => b.run.timestamp.localeCompare(a.run.timestamp));
}

/** The named run, or the newest one when no name is given. */
export function selectRun(file?: string): RunResult | null {
  const runs = listRuns();
  return (file ? runs.find((r) => r.run.file === file) : runs[0]) ?? null;
}

function read(file: string): RunResult {
  const result = JSON.parse(fs.readFileSync(file, "utf8")) as RunResult;
  result.run.stages ??= stagesFromLog(file);
  return result;
}

/** Older runs logged stage timings instead of writing them. Drop once they all do. */
function stagesFromLog(jsonPath: string) {
  let log: string;
  try {
    log = fs.readFileSync(jsonPath.replace(/\.json$/, ".log"), "utf8");
  } catch {
    return undefined;
  }

  // Last match wins: the log is appended to on a repeat run.
  const seconds = (stage: string) => {
    const found = [...log.matchAll(new RegExp(`${stage} done:.* in ([0-9.]+)s`, "g"))];
    return found.length ? Number(found[found.length - 1][1]) : undefined;
  };

  const ocr = seconds("OCR");
  const ner = seconds("NER");
  const norm = seconds("Norm");
  if (ocr === undefined || ner === undefined || norm === undefined) return undefined;

  return { ocr, ner, norm };
}
