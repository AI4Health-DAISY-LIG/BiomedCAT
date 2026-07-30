import fs from "node:fs";
import path from "node:path";
import { SUPPORTED, type QueuedDeck } from "./decks";
import { DATASET_DIR, OUTPUT_DIR, outputNameFor, readDir } from "./paths";

/** Decks in Dataset/ with no result yet: exactly what the next run will process. */
export function listQueued(): QueuedDeck[] {
  return readDir(DATASET_DIR)
    .filter((name) => SUPPORTED.includes(path.extname(name).toLowerCase()))
    .filter((name) => !hasResult(name))
    .map((name) => ({
      name,
      bytes: fs.statSync(path.join(DATASET_DIR, name)).size,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function hasResult(deck: string) {
  return fs.existsSync(path.join(OUTPUT_DIR, outputNameFor(deck)));
}
