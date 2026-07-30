import { FileJson, Sheet } from "lucide-react";

/** Downloads for one run. Plain links, so no client JavaScript is involved. */
export default function ExportLinks({ file }: { file: string }) {
  const href = (format: string) =>
    `/api/export?file=${encodeURIComponent(file)}&format=${format}`;

  return (
    <div className="flex gap-2">
      <Download href={href("json")} icon={FileJson} label="JSON" />
      <Download href={href("csv")} icon={Sheet} label="CSV" />
    </div>
  );
}

function Download({
  href,
  icon: Icon,
  label,
}: {
  href: string;
  icon: typeof FileJson;
  label: string;
}) {
  return (
    <a
      href={href}
      download
      className="flex h-10 items-center gap-2 rounded-xl border border-line bg-card px-4 text-body font-medium text-muted transition-colors duration-150 hover:border-line-strong hover:text-fg"
    >
      <Icon size={16} strokeWidth={1.75} />
      {label}
    </a>
  );
}
