import { Clock, Images, Tags } from "lucide-react";
import BarChart from "@/components/bar-chart";
import Card from "@/components/card";
import Dashboard from "@/components/dashboard";
import EmptyState from "@/components/empty-state";
import EntityTable from "@/components/entity-table";
import ExportLinks from "@/components/export-links";
import Meter from "@/components/meter";
import SlideText from "@/components/slide-text";
import StageTiming from "@/components/stage-timing";
import StatRow from "@/components/stat-row";
import { typeDistribution } from "@/lib/entity-types";
import { formatElapsed } from "@/lib/format";
import { selectRun } from "@/lib/runs";
import { stageTimings } from "@/lib/stages";
import { linkedCount, linkedPct } from "@/lib/stats";
import PipelineEventListener from "@/components/pipeline-event-listener";

// Read the output directory on every request, so a fresh pipeline run shows up
// without rebuilding.
export const dynamic = "force-dynamic";

export default async function Run({
  searchParams,
}: {
  searchParams: Promise<{ file?: string }>;
}) {
  // ?file= comes from the Overview table; without it, show the newest run.
  const { file } = await searchParams;
  const result = selectRun(file);

  if (!result) {
    return (
      <Dashboard title="Run">
        <EmptyState
          className="lg:col-span-3"
          message={
            file
              ? `No run found for ${file}.`
              : "No runs found in the output directory."
          }
        />
      </Dashboard>
    );
  }

  const { run, slides, entities } = result;
  const stages = stageTimings(run);

  return (
    <Dashboard title={run.file} actions={<ExportLinks file={run.file} />}>
      {/* The listener is injected here to monitor updates for the current view */}
      <PipelineEventListener />

      <Card
        title="Run summary"
        subtitle="Extraction totals"
        className="lg:col-span-2"
      >
        <StatRow
          columns={3}
          stats={[
            { icon: Images, value: slides.length, label: "Slides" },
            { icon: Tags, value: entities.length, label: "Entities" },
            {
              icon: Clock,
              value: formatElapsed(run.elapsed_s),
              label: "Elapsed",
            },
          ]}
        />

        {stages && (
          <div className="mt-6 border-t border-line pt-5">
            <p className="text-label uppercase tracking-[0.08em] text-muted">
              Time by stage
            </p>
            <div className="mt-4">
              <StageTiming stages={stages} />
            </div>
          </div>
        )}
      </Card>

      <Card title="Linking rate" subtitle="Proportion of entities normalized">
        <Meter
          pct={linkedPct(entities)}
          caption={`${linkedCount(entities)} of ${entities.length} entities resolved to an identifier.`}
        />
      </Card>

      <Card
        title="Entities"
        subtitle="Extracted terms with assigned identifiers"
        className="h-full lg:col-span-2"
        flush
      >
        <EntityTable entities={entities} />
      </Card>

      <Card title="Entity types" subtitle="Counts by type" className="h-full">
        <BarChart data={typeDistribution(entities)} />
      </Card>

      <Card
        title="OCR text"
        subtitle="Text extracted per slide"
        className="lg:col-span-3"
      >
        <SlideText slides={slides} />
      </Card>
    </Dashboard>
  );
}
