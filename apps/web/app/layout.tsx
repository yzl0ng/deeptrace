import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import "./globals.css";
import "./language.css";

const sans = Manrope({ variable: "--font-sans", subsets: ["latin"] });
const mono = IBM_Plex_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400", "500", "600"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://deeptrace-r1.example.com"),
  title: { default: "DeepTrace-R1 — Inspectable Research Agent", template: "%s — DeepTrace-R1" },
  description: "An inspectable research-agent system with verified execution traces, evidence ledgers and held-out evaluation.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: { title: "DeepTrace-R1", description: "See the research agent think in evidence.", images: ["/og.svg"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${sans.variable} ${mono.variable}`}>{children}</body></html>;
}
