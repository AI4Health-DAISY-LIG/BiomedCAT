import fs from "node:fs";
import path from "node:path";

// The pipeline reads decks from Dataset/ and writes results to Output/, both
// siblings of the frontend. BIOMEDCAT_ROOT overrides that layout.
const root = process.env.BIOMEDCAT_ROOT ?? path.join(process.cwd(), "..");

export const DATASET_DIR = path.join(root, "Dataset");
export const OUTPUT_DIR = path.join(root, "Output");

/** Output filename the pipeline writes for a given deck. */
export function outputNameFor(deck: string) {
  return `${path.parse(deck).name}_BiomedCAT.json`;
}

/** Entries, or none: neither directory exists before the first run. */
export function readDir(dir: string): string[] {
  try {
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}
