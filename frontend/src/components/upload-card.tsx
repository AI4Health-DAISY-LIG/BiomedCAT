"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, UploadCloud } from "lucide-react";
import Card from "@/components/card";
import { formatBytes } from "@/lib/format";
import { SUPPORTED, type QueuedDeck } from "@/lib/decks";

export default function UploadCard({
  queued,
  className,
}: {
  queued: QueuedDeck[];
  className?: string;
}) {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    setBusy(true);
    setError(null);

    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body });

    if (!res.ok) {
      const { error } = await res.json().catch(() => ({ error: "Upload failed." }));
      setError(error);
    } else {
      router.refresh(); // pull the new deck into the queue below
    }
    setBusy(false);
  }

  return (
    <Card
      title="Add a deck"
      subtitle="Queued for the next pipeline run"
      className={className}
    >
      <div className="flex h-full flex-col gap-4">
        <div
          onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) upload(file);
          }}
          onClick={() => input.current?.click()}
          className={`flex min-h-[150px] flex-1 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 text-center transition-colors ${
            dragging
              ? "border-accent bg-accent/10"
              : "border-line-strong bg-inset hover:border-muted"
          }`}
        >
          {busy ? (
            <Loader2 size={26} className="animate-spin text-accent" />
          ) : (
            <UploadCloud size={26} strokeWidth={1.5} className="text-accent" />
          )}
          <p className="mt-4 text-lead font-semibold">
            {busy ? "Uploading…" : "Drop a deck here or browse"}
          </p>
          <p className="mt-2 text-label text-muted">{SUPPORTED.join("  ·  ")}</p>

          <input
            ref={input}
            type="file"
            accept={SUPPORTED.join(",")}
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) upload(file);
              e.target.value = ""; // let the same file be picked again after an error
            }}
          />
        </div>

        {error && <p className="text-label text-danger">{error}</p>}

        {queued.length > 0 && (
          <ul className="flex flex-col gap-2">
            {queued.map((deck) => (
              <li
                key={deck.name}
                className="flex items-center justify-between gap-3 rounded-lg border border-line bg-inset px-3 py-2"
              >
                <span className="truncate text-body">{deck.name}</span>
                <span className="shrink-0 text-label text-muted">
                  {formatBytes(deck.bytes)} · queued
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
