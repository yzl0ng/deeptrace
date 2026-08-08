import type { Metadata } from "next";
import { ArchitectureDetail } from "../../../architecture-detail";

export const metadata: Metadata = { title: "架构模块工程详情" };

export default async function ArchitectureDetailPageZh({ searchParams }: { searchParams: Promise<{ node?: string }> }) {
  const params = await searchParams;
  return <ArchitectureDetail lang="zh" nodeId={params.node}/>;
}
