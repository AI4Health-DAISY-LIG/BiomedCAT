/** The dashboard's only container: a titled panel. Every block sits in one. */
export default function Card({
  title,
  subtitle,
  className,
  flush,
  children,
}: {
  title?: string;
  subtitle?: string;
  className?: string;
  /** For tables: their cells carry the horizontal padding. */
  flush?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section
      className={`flex min-w-0 flex-col rounded-2xl border border-line bg-card ${className ?? ""}`}
    >
      {title && (
        <header className="px-6 pt-6">
          <h3 className="font-display text-heading font-bold tracking-[-0.015em]">{title}</h3>
          {subtitle && <p className="mt-1 text-label text-muted">{subtitle}</p>}
        </header>
      )}
      <div className={`flex-1 ${flush ? "pb-2 pt-4" : "p-6"}`}>{children}</div>
    </section>
  );
}
