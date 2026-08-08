import type { Metadata } from "next";
import { ArchitectureMap } from "../architecture-map";

export const metadata: Metadata = { title: "Interactive Engineering Map" };

export default function ArchitecturePage() {
  return <ArchitectureMap lang="en"/>;
}
