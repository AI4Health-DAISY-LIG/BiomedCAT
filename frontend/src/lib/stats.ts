import type { Entity, RunResult } from "./runs";

/** How many entities Norm linked to a CURIE. */
export function linkedCount(entities: Entity[]) {
  return entities.filter((e) => e.curie !== null).length;
}

/** Share of entities linked, as a whole percent. */
export function linkedPct(entities: Entity[]) {
  if (entities.length === 0) return 0;
  return Math.round((linkedCount(entities) / entities.length) * 100);
}

/** What the Overview leads with, summed across every run. */
export function batchTotals(runs: RunResult[]) {
  return {
    decks: runs.length,
    slides: runs.reduce((n, r) => n + r.slides.length, 0),
    entities: runs.flatMap((r) => r.entities),
  };
}
