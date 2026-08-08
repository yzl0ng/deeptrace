import type { Metadata } from "next";
import { ArchitectureMap } from "../../architecture-map";

export const metadata: Metadata = { title: "可交互工程架构导图" };

export default function ArchitecturePageZh() {
  return <ArchitectureMap lang="zh"/>;
}
