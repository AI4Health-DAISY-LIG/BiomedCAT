import { ChevronRight } from "lucide-react";
import type { Slide } from "@/lib/runs";

/** One collapsed row per slide. <details> keeps this off the client bundle. */
export default function SlideText({ slides }: { slides: Slide[] }) {
  return (
    <div className="flex flex-col gap-3">
      {slides.map((slide) => (
        <details
          key={slide.page}
          className="group rounded-xl border border-line bg-inset"
        >
          <summary className="flex cursor-pointer list-none items-center gap-3 rounded-xl px-5 py-4 transition-colors duration-150 hover:bg-white/[0.04] [&::-webkit-details-marker]:hidden">
            <ChevronRight
              size={16}
              className="shrink-0 text-muted transition-transform group-open:rotate-90"
            />
            <span className="text-body font-semibold">Slide {slide.page}</span>
          </summary>

          <p className="max-w-[100ch] whitespace-pre-wrap px-5 pb-5 pl-12 text-lead text-muted">
            {slide.text}
          </p>
        </details>
      ))}
    </div>
  );
}
