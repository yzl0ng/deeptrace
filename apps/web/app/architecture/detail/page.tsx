import type { Metadata } from "next";
import { ArchitectureDetail } from "../../architecture-detail";

export const metadata: Metadata = { title: "Architecture Module Detail" };

export default async function ArchitectureDetailPage({ searchParams }: { searchParams: Promise<{ node?: string }> }) {
  const params = await searchParams;
  return <ArchitectureDetail lang="en" nodeId={params.node}/>;
}
