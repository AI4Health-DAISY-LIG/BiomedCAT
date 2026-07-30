import type { Metadata } from "next";
import { JetBrains_Mono, Plus_Jakarta_Sans, Space_Grotesk } from "next/font/google";
import Sidebar from "@/components/sidebar";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  variable: "--font-jakarta",
  subsets: ["latin"],
});

// Titles get their own voice rather than just a larger size.
const display = Space_Grotesk({
  variable: "--font-space",
  subsets: ["latin"],
});

// A CURIE is read character by character, so identifiers get a monospace face.
const mono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BiomedCAT",
  description: "Biomedical entity extraction and normalization from slide decks.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${jakarta.variable} ${display.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-ink font-sans text-fg">
        {/* Rail beside the content on desktop, banner above it on narrow screens. */}
        <div className="flex min-h-screen flex-col lg:flex-row">
          <Sidebar />
          <main className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8 2xl:px-12">
            {/* The cap only bites on very large displays. */}
            <div className="mx-auto w-full max-w-[2200px]">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
