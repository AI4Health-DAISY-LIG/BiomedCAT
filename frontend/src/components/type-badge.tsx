import { formatType } from "@/lib/format";
import { typeColor } from "@/lib/entity-types";

/**
 * An entity type wherever it appears. The dot carries the colour and the label
 * carries the meaning, so the type is never identified by colour alone.
 */
export default function TypeBadge({ type }: { type: string }) {
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap rounded-md border border-line bg-inset py-1.5 pl-2.5 pr-3 text-label">
      <span
        className="h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: typeColor(type) }}
      />
      {formatType(type)}
    </span>
  );
}
