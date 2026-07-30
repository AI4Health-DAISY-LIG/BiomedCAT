"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Info, LayoutDashboard, Play } from "lucide-react";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/run", label: "Run", icon: Play },
  { href: "/about", label: "About", icon: Info },
];

export default function Sidebar() {
  const pathname = usePathname();

  // self-start stops the flex row stretching the rail, which is what gives
  // sticky room to work; h-screen then pins it for the whole scroll.
  return (
    <aside className="flex shrink-0 flex-col border-b border-line bg-panel lg:sticky lg:top-0 lg:h-screen lg:w-60 lg:self-start lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="flex items-center px-5 pt-5 lg:h-[76px] lg:px-6 lg:pt-0">
        <h1 className="font-display text-[26px] font-bold leading-none tracking-[-0.03em]">
          Biomed<span className="text-accent">CAT</span>
        </h1>
      </div>

      {/* Stacked rail on desktop; a scrollable row of pills on narrow screens. */}
      <nav className="flex gap-2 overflow-x-auto px-4 py-4 lg:flex-col lg:gap-1.5 lg:py-0">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`flex h-11 shrink-0 items-center gap-3 rounded-xl px-3.5 text-body transition-colors duration-150 ${
                active
                  ? "bg-accent font-semibold text-ink"
                  : "font-medium text-muted hover:bg-white/5 hover:text-fg"
              }`}
            >
              <Icon
                size={18}
                strokeWidth={1.75}
                fill={active ? "currentColor" : "none"}
              />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
