import { Fragment } from "react";
import Image from "next/image";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { ENTITY_TYPES, TYPE_DEFINITIONS, typeColor } from "@/lib/entity-types";
import { SUPPORTED } from "@/lib/decks";
import { formatType } from "@/lib/format";

/** Editorial layout rather than the dashboard grid: one idea per band. */
export default function About() {
  return (
    <div className="mx-auto max-w-[1400px]">
      <Hero />
      <Facts />
      <Rail />
      <Motivation />
      <Vocabulary />
      <Grounding />
      <Console />
    </div>
  );
}

function Hero() {
  return (
    <section className="relative grid gap-12 py-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:items-center lg:py-16">
      <div
        aria-hidden
        className="pointer-events-none absolute -left-40 top-0 h-[460px] w-[720px] rounded-full bg-accent/10 blur-[140px]"
      />

      <div className="relative">
        <h1 className=" font-display text-display font-bold tracking-[-0.035em]">
          Slides in.
          <br />
          Grounded entities out.
        </h1>

        <p className="mt-6 max-w-xl text-lead text-muted">
          BiomedCAT reads the text off each slide of a presentation, extracts the
          typed biomedical entities, and links every one of them to a standard
          identifier in a biomedical knowledge base.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Link
            href="/"
            className="flex h-11 items-center gap-2 rounded-xl bg-accent px-5 text-body font-semibold text-ink transition-opacity duration-150 hover:opacity-90"
          >
            Open the console
            <ArrowRight size={17} strokeWidth={2} />
          </Link>
          <Link
            href="/run"
            className="flex h-11 items-center rounded-xl border border-line bg-card px-5 text-body font-medium text-muted transition-colors duration-150 hover:border-line-strong hover:text-fg"
          >
            View a run
          </Link>
        </div>
      </div>

      {/* Light figure framed as a plate, the way it sits in a paper. */}
      <figure className="relative">
        <div className="rounded-2xl border border-line bg-[#eef2f5] p-3">
          <Image
            src="/media/pipeline.png"
            alt="The BiomedCAT pipeline: slides enter OCR, text passes to named entity recognition, entities are normalized to CURIEs."
            width={2752}
            height={1536}
            sizes="(max-width: 1024px) 100vw, 720px"
            priority
            className="rounded-lg"
          />
        </div>
        <figcaption className="mt-3 text-label text-muted">
          Slide decks in, one identifier per entity out.
        </figcaption>
      </figure>
    </section>
  );
}

const FACTS = [
  { value: "3", label: "Pipeline stages" },
  { value: "7", label: "Entity types" },
  { value: "2", label: "Knowledge bases" },
  { value: String(SUPPORTED.length), label: "Input formats" },
];

function Facts() {
  return (
    <section className="grid border-t border-line sm:grid-cols-2 lg:grid-cols-4">
      {FACTS.map((fact) => (
        <div
          key={fact.label}
          className="border-line py-8 sm:[&:nth-child(n+3)]:border-t lg:border-l lg:py-10 lg:first:border-l-0 lg:[&:nth-child(n+3)]:border-t-0 lg:[&:not(:first-child)]:pl-8"
        >
          <p className="font-display text-hero font-bold tracking-[-0.04em]">
            {fact.value}
          </p>
          <p className="mt-2 text-label uppercase tracking-[0.12em] text-muted">
            {fact.label}
          </p>
        </div>
      ))}
    </section>
  );
}

// Artifacts are real values from the FSHD1 run, not placeholders.
const STAGES = [
  {
    name: "Read",
    stage: "OCR",
    summary:
      "A vision-language model transcribes each slide, keeping reading order across columns and figure labels.",
    model: "zai-org/GLM-OCR",
    artifact: (
      <p className="text-body leading-relaxed text-muted">
        Hypomethylation of D4Z4 repeats in chromosome 4 results in production of
        toxic DUX4 protein.
      </p>
    ),
  },
  {
    name: "Extract",
    stage: "NER",
    summary:
      "A general instruction model marks the biomedical terms and assigns each one a type, with no task-specific training.",
    model: "meta-llama/Llama-3.1-8B-Instruct",
    artifact: (
      <div className="flex flex-wrap gap-2">
        {[
          ["DUX4 protein", "GENE"],
          ["D4Z4 repeats", "CHROMOSOMAL_LOCUS"],
          ["chromosome 4", "CHROMOSOMAL_LOCUS"],
        ].map(([text, type]) => (
          <span
            key={text}
            className="inline-flex items-center gap-2 rounded-md border border-line bg-card py-1.5 pl-2.5 pr-3 text-label"
          >
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: typeColor(type) }}
            />
            {text}
          </span>
        ))}
      </div>
    ),
  },
  {
    name: "Ground",
    stage: "Normalization",
    summary:
      "Candidates are retrieved from two public resolvers, then a judge selects the type-consistent identifier or abstains.",
    model: "meta-llama/Llama-3.1-8B-Instruct",
    artifact: (
      <ul className="flex flex-col gap-1.5 font-mono text-label text-accent">
        <li>NCBIGene:100288687</li>
        <li>UMLS:C5542200</li>
        <li>MESH:D002894</li>
      </ul>
    ),
  },
];

function Rail() {
  return (
    <Section
      title="Pipeline"
      lead="Each stage is an independent module with a single entry point that manages its own model lifecycle, so a stage can be replaced without touching the others. What each one emits is shown below, using values from a real run."
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-stretch">
        {STAGES.map((stage, i) => (
          <Fragment key={stage.stage}>
            {i > 0 && (
              <div className="flex shrink-0 items-center justify-center py-1 lg:py-0">
                <ArrowRight
                  size={18}
                  className="rotate-90 text-muted/50 lg:rotate-0"
                />
              </div>
            )}
            <article className="flex flex-1 flex-col rounded-2xl border border-line bg-card p-6">
              <div className="flex items-baseline gap-3">
                <span className="font-display text-label font-bold tracking-[0.1em] text-accent">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <h3 className="font-display text-heading font-bold tracking-[-0.015em]">
                  {stage.name}
                </h3>
                <span className="text-label uppercase tracking-[0.08em] text-muted">
                  {stage.stage}
                </span>
              </div>

              <p className="mt-3 text-body text-muted">{stage.summary}</p>

              <div className="mt-6 flex-1 rounded-xl border border-line bg-inset p-4">
                {stage.artifact}
              </div>

              <p className="mt-4 break-all font-mono text-label text-muted">
                {stage.model}
              </p>
            </article>
          </Fragment>
        ))}
      </div>
    </Section>
  );
}

function Motivation() {
  return (
    <Section
      title="Motivation"
      lead="Knowledge-graph construction methods assume clean plain text such as PubMed abstracts. Much of what a laboratory knows never arrives in that form."
    >
      <div className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)] lg:items-center">
        <figure className="relative overflow-hidden rounded-2xl border border-line">
          <Image
            src="/media/drugs.png"
            alt="A gloved hand pipetting culture medium into a six-well plate at a laboratory bench."
            width={1440}
            height={960}
            sizes="(max-width: 1024px) 100vw, 760px"
            loading="eager"
            className="w-full object-cover"
          />
          {/* Scrim so the photograph sits inside a dark page. */}
          <div
            aria-hidden
            className="absolute inset-0 bg-gradient-to-t from-ink/70 via-ink/15 to-transparent"
          />
        </figure>

        <div className="space-y-5">
          <p className="text-lead text-muted">
            Results are produced at the bench and communicated in slides, where
            text sits beside figures, tables, and charts. A single corrupted
            character invalidates a gene symbol, and every stage downstream
            inherits the error.
          </p>
          <p className="text-lead text-muted">
            BiomedCAT treats input modality as a first-class problem rather than
            a preprocessing detail. Each stage is measured on its own output, so
            a failure can be attributed to transcription, to extraction, or to
            grounding.
          </p>
          <p className="text-lead text-muted">
            The working domain is facioscapulohumeral muscular dystrophy, a
            disease whose literature turns on a small, precise vocabulary of
            genes, loci, and epigenetic marks.
          </p>
        </div>
      </div>
    </Section>
  );
}

function Vocabulary() {
  return (
    <Section
      title="Entity types"
      lead="One label set is shared by extraction, typing, and normalization, so no stage can drift on what a type means. The same type also constrains retrieval, and keeps one colour everywhere it appears in the console."
    >
      <ul className="grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
        {ENTITY_TYPES.map((type) => (
          <li key={type} className="border-t border-line pt-4">
            <div className="flex items-center gap-2.5">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: typeColor(type) }}
              />
              <span className="text-body font-semibold">
                {formatType(type)}
              </span>
            </div>
            <p className="mt-1.5 text-label text-muted">
              {TYPE_DEFINITIONS[type]}
            </p>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function Grounding() {
  return (
    <Section
      title="Grounding"
      lead="Linking on string similarity alone attaches confident-looking identifiers to the wrong concept. Retrieval and selection are kept as separate steps so the failure modes stay separable."
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Retrieval">
          <ul className="flex flex-col gap-3">
            {[
              ["RENCI name resolution", "name-resolution-sri.renci.org"],
              ["ARAX entity normalizer", "arax.ncats.io"],
            ].map(([name, host]) => (
              <li key={host}>
                <p className="text-body">{name}</p>
                <p className="mt-0.5 font-mono text-label text-muted">{host}</p>
              </li>
            ))}
          </ul>
          <p className="mt-5 text-body text-muted">
            An unconstrained lookup in both resolvers runs alongside one
            type-constrained pass per entity type. Up to ten candidates come back
            from each, merged into a single pool.
          </p>
        </Panel>

        <Panel title="Selection">
          <p className="text-body text-muted">
            The pool is put to a language model as a numbered menu, each option
            carrying its knowledge-base type, so the choice can be checked
            against the type the entity was assigned.
          </p>
          <p className="mt-3 text-body text-muted">
            Option zero is always <em className="not-italic text-fg">none</em>.
            An abstention is recorded as an unlinked entity rather than forced
            onto the nearest string match, and is kept distinct from a reply the
            parser could not read.
          </p>
        </Panel>

        <Panel title="Reproducibility">
          <p className="text-body text-muted">
            Decoding is greedy at every stage, so the same deck yields the same
            output when the pipeline is run again.
          </p>
          <p className="mt-3 text-body text-muted">
            Each result file records the models and resolver endpoints that
            produced it, alongside the elapsed time, so a figure can be traced
            back to the configuration that generated it.
          </p>
        </Panel>
      </div>
    </Section>
  );
}

function Console() {
  return (
    <Section
      title="Console"
      lead="This interface reads what the pipeline has already written. It does not run models."
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.8fr)]">
        <Panel title="Adding a deck">
          <p className="text-body text-muted">
            Decks added in Overview are written to <Path>Dataset/</Path> and
            picked up by the next pipeline run, which skips anything already
            processed. Results are read from <Path>Output/</Path>.
          </p>
          <p className="mt-3 text-body text-muted">
            A run requires a CUDA GPU and takes several minutes per deck, so it
            is started from the command line, not the browser.
          </p>
        </Panel>

        <Panel title="Taking results away">
          <p className="text-body text-muted">
            Every run downloads from its own page as the pipeline&apos;s original
            JSON, or as a CSV of entities with their types and identifiers.
          </p>
          <p className="mt-3 text-body text-muted">
            The CSV quotes every field and carries a byte-order mark, so slide
            text containing commas, quotes, or line breaks survives a spreadsheet
            import intact.
          </p>
        </Panel>

        <Panel title="Start here">
          <p className="text-body text-muted">
            The Overview lists every processed deck with its linking rate. Each
            row opens the full extraction for that deck.
          </p>
          <Link
            href="/"
            className="mt-5 inline-flex h-11 items-center gap-2 rounded-xl bg-accent px-5 text-body font-semibold text-ink transition-opacity duration-150 hover:opacity-90"
          >
            Open the console
            <ArrowRight size={17} strokeWidth={2} />
          </Link>
        </Panel>
      </div>
    </Section>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-line bg-card p-6">
      <h3 className="font-display text-heading font-bold tracking-[-0.015em]">
        {title}
      </h3>
      <div className="mt-4">{children}</div>
    </div>
  );
}

function Path({ children }: { children: React.ReactNode }) {
  return <code className="font-mono text-label text-fg">{children}</code>;
}

function Section({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-line py-14">
      <h2 className="font-display text-title font-bold tracking-[-0.025em]">
        {title}
      </h2>
      {lead && <p className="mt-4 max-w-4xl text-lead text-muted">{lead}</p>}

      <div className="mt-10">{children}</div>
    </section>
  );
}
