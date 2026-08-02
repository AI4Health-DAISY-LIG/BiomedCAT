import { FileStack, Images, Link2, Tags } from "lucide-react";
import Card from "@/components/card";
import Dashboard from "@/components/dashboard";
import EmptyState from "@/components/empty-state";
import RunsTable from "@/components/runs-table";
import StatRow from "@/components/stat-row";
import UploadCard from "@/components/upload-card";
import { listQueued } from "@/lib/queue";
import { listRuns } from "@/lib/runs";
import { batchTotals, linkedCount, linkedPct } from "@/lib/stats";
import PipelineEventListener from "@/components/pipeline-event-listener";

export const dynamic = "force-dynamic";

// The batch: what is queued, what has been processed. Anything about a single
// deck's entities belongs on /run, so the two pages never show the same number.
export default function Home() {
  const runs = listRuns();
  const { decks, slides, entities } = batchTotals(runs);

  return (
    <Dashboard title="Overview" subtitle="Processed decks and aggregate totals.">
      {/* The listener is injected here to monitor global pipeline events */}
      <PipelineEventListener />

      {/* First, because adding a deck is what a visitor came to do. */}
      <UploadCard queued={listQueued()} className="h-full" />

      <Card
        title="Batch summary"
        subtitle="Totals across all runs"
        className="h-full lg:col-span-2"
      >
        <StatRow
          stats={[
            { icon: FileStack, value: decks, label: "Decks" },
            { icon: Images, value: slides, label: "Slides" },
            { icon: Tags, value: entities.length, label: "Entities" },
            {
              icon: Link2,
              value: `${linkedPct(entities)}%`,
              label: "Linked to a CURIE",
              note: `${linkedCount(entities)} of ${entities.length}`,
            },
          ]}
        />
      </Card>

      {runs.length > 0 ? (
        <Card title="Runs" subtitle="Ordered by date, newest first" className="lg:col-span-3" flush>
          <RunsTable runs={runs} />
        </Card>
      ) : (
        <EmptyState
          className="lg:col-span-3"
          message="No runs recorded. Add a deck, then execute the pipeline."
        />
      )}
    </Dashboard>
  );
}
