import type { RunResult } from "./runs";

/**
 * Three hues, not one hue stepped light to dark: in a bar this thin, lightness
 * alone is not enough separation. Kept distinct from the entity type palette,
 * which appears beside this on the run page.
 */
const COLORS = ["#a7dfd3", "#e58f72", "#bf72e5"];
const LABELS = ["OCR", "NER", "Normalization"];

export type StageTiming = {
  label: string;
  seconds: number;
  color: string;
};

/** Null when the run has no stage timings, so callers can omit the chart. */
export function stageTimings(run: RunResult["run"]): StageTiming[] | null {
  if (!run.stages) return null;

  const seconds = [run.stages.ocr, run.stages.ner, run.stages.norm];
  if (seconds.some((s) => typeof s !== "number")) return null;

  return LABELS.map((label, i) => ({
    label,
    seconds: seconds[i],
    color: COLORS[i],
  }));
}
