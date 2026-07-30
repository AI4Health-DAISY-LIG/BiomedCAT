/** A title over a three-column grid. items-start stops short cards stretching. */
export default function Dashboard({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-5 2xl:gap-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-title font-bold tracking-[-0.02em]">
            {title}
          </h2>
          {subtitle && <p className="mt-2 text-lead text-muted">{subtitle}</p>}
        </div>
        {actions}
      </header>

      <div className="grid items-start gap-5 lg:grid-cols-3 2xl:gap-6">{children}</div>
    </div>
  );
}
