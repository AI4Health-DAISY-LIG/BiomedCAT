import Card from "@/components/card";

export default function EmptyState({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <p className="py-10 text-center text-lead text-muted">{message}</p>
    </Card>
  );
}
