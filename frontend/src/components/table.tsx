/** Shared table shell, so every table on the dashboard has the same rhythm. */
export function Table({
  headers,
  maxHeight,
  children,
}: {
  headers: string[];
  /** Caps a long table and scrolls it, keeping the header pinned. */
  maxHeight?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={maxHeight ? "overflow-auto" : "overflow-x-auto"}
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full text-left text-body">
        <thead className="text-label uppercase tracking-[0.06em] text-muted">
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                className="sticky top-0 border-b border-line bg-card px-6 py-3.5 font-semibold"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({ children }: { children: React.ReactNode }) {
  return (
    <tr className="border-b border-line transition-colors duration-150 last:border-0 hover:bg-white/[0.02]">
      {children}
    </tr>
  );
}

export function Cell({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={`px-6 py-3.5 ${className ?? ""}`}>{children}</td>;
}
