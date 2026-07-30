import { formatType } from "./format";
import type { Entity } from "./runs";

/**
 * One colour per entity type, in the order biomedcat/types.py declares them.
 * The order is the colour-blind-safety mechanism: adjacent slots were checked
 * for separation, so a type keeps its slot and charts keep this order rather
 * than sorting by count. Colour is always shown beside the type name.
 */
const TYPE_COLORS: Record<string, string> = {
  GENE: "#3987e5",
  DISEASE: "#d95926",
  CHEMICAL: "#199e70",
  CELL_TYPE: "#c98500",
  ANATOMY: "#d55181",
  CHROMOSOMAL_LOCUS: "#008300",
  EPIGENETIC_MODIFICATION: "#9085e9",
};

/** The gloss each type carries in biomedcat/types.py, kept verbatim. */
export const TYPE_DEFINITIONS: Record<string, string> = {
  GENE: "a gene or gene symbol",
  DISEASE: "a disease or disorder",
  CHEMICAL: "a chemical, drug, or compound",
  CELL_TYPE: "a type of cell",
  ANATOMY: "an anatomical structure, tissue, or organ",
  CHROMOSOMAL_LOCUS: "a chromosomal location, locus, or genomic region",
  EPIGENETIC_MODIFICATION: "an epigenetic modification such as methylation",
};

export const ENTITY_TYPES = Object.keys(TYPE_COLORS);

/** Muted ink for a type the pipeline adds later. */
export function typeColor(type: string) {
  return TYPE_COLORS[type] ?? "#8b9599";
}

/** Counts as chart bars. Always all seven: a missing type is a zero, not a gap. */
export function typeDistribution(entities: Entity[]) {
  return ENTITY_TYPES.map((type) => ({
    label: formatType(type),
    value: entities.filter((e) => e.type === type).length,
    color: typeColor(type),
  }));
}
