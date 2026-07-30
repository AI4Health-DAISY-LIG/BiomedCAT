import type { LucideIcon } from "lucide-react";

export type Stat = {
  icon: LucideIcon;
  value: string | number;
  label: string;
  note?: string;
};

const COLUMNS = {
  2: "sm:grid-cols-2",
  3: "sm:grid-cols-3",
  4: "sm:grid-cols-4",
};

/** The KPI row both pages open with. */
export default function StatRow({
  stats,
  columns = 4,
}: {
  stats: Stat[];
  columns?: keyof typeof COLUMNS;
}) {
  return (
    <div className={`grid grid-cols-2 gap-4 ${COLUMNS[columns]}`}>
      {stats.map(({ icon: Icon, value, label, note }) => (
        <div key={label} className="rounded-xl border border-line bg-inset p-5">
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <Icon size={19} strokeWidth={1.75} />
          </span>
          <p className="mt-4 font-display text-3xl font-bold tracking-[-0.03em]">{value}</p>
          <p className="mt-1.5 text-label text-muted">{label}</p>
          {note && <p className="mt-1 text-label text-muted/80">{note}</p>}
        </div>
      ))}
    </div>
  );
}
