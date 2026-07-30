import fs from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { SUPPORTED } from "@/lib/decks";
import { DATASET_DIR, outputNameFor } from "@/lib/paths";

/**
 * Drops a deck into Dataset/ for the pipeline's next run. Nothing is processed
 * here: the pipeline is a batch script, not a service.
 */
export async function POST(request: Request) {
  const form = await request.formData();
  const file = form.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "No file received." }, { status: 400 });
  }

  // Trust nothing from the filename but its last segment.
  const name = path.basename(file.name);
  const ext = path.extname(name).toLowerCase();

  if (!SUPPORTED.includes(ext)) {
    return NextResponse.json(
      { error: `Unsupported format ${ext || "(none)"}. Accepts ${SUPPORTED.join(", ")}.` },
      { status: 415 },
    );
  }

  const target = path.join(DATASET_DIR, name);
  await fs.mkdir(DATASET_DIR, { recursive: true });

  // Refuse to clobber a deck that is already queued or already processed.
  if (await exists(target)) {
    return NextResponse.json(
      { error: `A deck named ${name} already exists in Dataset/.` },
      { status: 409 },
    );
  }

  await fs.writeFile(target, Buffer.from(await file.arrayBuffer()));

  return NextResponse.json({ name, output: outputNameFor(name) }, { status: 201 });
}

async function exists(p: string) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}
