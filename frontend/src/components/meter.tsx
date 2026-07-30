/**
 * A single ratio against its limit: hero figure plus a same-ramp track.
 * `size="sm"` drops the figure, for use inside a table row.
 */
export default function Meter({
  pct,
  caption,
  size = "lg",
}: {
  pct: number;
  caption?: string;
  size?: "sm" | "lg";
}) {
  const bar = (
    <span className={`block ${size === "lg" ? "h-3" : "h-1.5"} rounded-[3px] bg-track`}>
      <span
        className="block h-full rounded-r-[4px] bg-accent"
        style={{ width: `${pct}%` }}
      />
    </span>
  );

  if (size === "sm") return bar;

  return (
    <div>
      <p className="font-display text-hero font-bold tracking-[-0.04em]">{pct}%</p>
      <div className="mt-4">{bar}</div>
      {caption && <p className="mt-4 text-label text-muted">{caption}</p>}
    </div>
  );
}
