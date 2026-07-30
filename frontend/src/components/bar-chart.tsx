export type Bar = {
  label: string;
  value: number;
  color: string;
};

/** Rows keep the order given: the colour order is what keeps hues distinguishable. */
export default function BarChart({ data }: { data: Bar[] }) {
  const max = Math.max(...data.map((d) => d.value), 1);

  return (
    <ul className="flex h-full flex-col justify-between gap-3">
      {data.map((d) => (
        <li
          key={d.label}
          className="grid grid-cols-[10.5rem_1fr] items-center gap-4"
        >
          <span className="truncate text-label text-muted" title={d.label}>
            {d.label}
          </span>

          {/* Full-width rail on every row, so a zero reads as measured, not missing. */}
          <div className="flex h-2.5 items-center rounded-[3px] bg-white/[0.05]">
            <span
              className="h-full rounded-[3px]"
              style={{
                width: `${(d.value / max) * 100}%`,
                backgroundColor: d.color,
              }}
            />
            <span
              className={`ml-3 text-label tabular-nums ${
                d.value === 0 ? "text-muted" : "font-medium"
              }`}
            >
              {d.value}
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}
