import { formatElapsed } from "@/lib/format";
import type { StageTiming as Timing } from "@/lib/stages";

/** Where a run spent its time. Segments are separated by a gap, not a stroke. */
export default function StageTiming({ stages }: { stages: Timing[] }) {
  const total = stages.reduce((n, s) => n + s.seconds, 0) || 1;
  const share = (s: Timing) => (s.seconds / total) * 100;

  return (
    <div>
      <div className="flex h-3.5 gap-[2px]">
        {stages.map((s) => (
          <span
            key={s.label}
            className="block first:rounded-l-[3px] last:rounded-r-[3px]"
            style={{ width: `${share(s)}%`, backgroundColor: s.color }}
          />
        ))}
      </div>

      <ul className="mt-3 flex flex-wrap gap-x-7 gap-y-2">
        {stages.map((s) => (
          <li key={s.label} className="flex items-center gap-2 text-label">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: s.color }}
            />
            <span className="text-muted">{s.label}</span>
            <span className="tabular-nums">{formatElapsed(s.seconds)}</span>
            <span className="tabular-nums text-muted">
              {Math.round(share(s))}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
