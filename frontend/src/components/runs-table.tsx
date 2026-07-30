import Link from "next/link";
import Meter from "@/components/meter";
import { Cell, Row, Table } from "@/components/table";
import { formatDate } from "@/lib/format";
import type { RunResult } from "@/lib/runs";
import { linkedPct } from "@/lib/stats";

export default function RunsTable({ runs }: { runs: RunResult[] }) {
  return (
    <Table headers={["#", "Deck", "Date", "Slides", "Entities", "Linked"]}>
      {runs.map((r, i) => (
        <Row key={r.run.file}>
          <Cell className="text-muted tabular-nums">
            {String(i + 1).padStart(2, "0")}
          </Cell>
          <Cell>
            <Link
              href={`/run?file=${encodeURIComponent(r.run.file)}`}
              className="font-medium transition-colors duration-150 hover:text-accent"
            >
              {r.run.file}
            </Link>
          </Cell>
          <Cell className="text-muted tabular-nums">
            {formatDate(r.run.timestamp)}
          </Cell>
          <Cell className="tabular-nums">{r.slides.length}</Cell>
          <Cell className="tabular-nums">{r.entities.length}</Cell>
          <Cell>
            <div className="flex min-w-[7rem] items-center gap-3">
              <span className="flex-1">
                <Meter pct={linkedPct(r.entities)} size="sm" />
              </span>
              <span className="w-10 text-right text-label font-medium tabular-nums">
                {linkedPct(r.entities)}%
              </span>
            </div>
          </Cell>
        </Row>
      ))}
    </Table>
  );
}
