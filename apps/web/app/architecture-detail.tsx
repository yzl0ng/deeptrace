import Link from "next/link";
import { architectureNodes, getArchitectureNode, type ArchitectureLanguage } from "./architecture-data";
import { SiteFooter, SiteHeader } from "./site-shell";

const labels = {
  en: {
    back: "BACK TO SYSTEM MAP", layer: "SYSTEM MODULE", input: "INPUT", process: "WHAT HAPPENS", output: "OUTPUT",
    real: "REAL IMPLEMENTATION", guards: "DETERMINISTIC GUARDS & FAILURE PATHS", proof: "EVIDENCE FROM THE PROJECT",
    prev: "PREVIOUS MODULE", next: "NEXT MODULE", map: "ALL MODULES", switch: "中文详情",
  },
  zh: {
    back: "返回系统总导图", layer: "系统模块", input: "输入", process: "这一步具体做什么", output: "输出",
    real: "真实工程实现", guards: "确定性护栏与异常路径", proof: "项目中的真实证据",
    prev: "上一个模块", next: "下一个模块", map: "全部模块", switch: "English detail",
  },
};

export function ArchitectureDetail({ lang, nodeId }: { lang: ArchitectureLanguage; nodeId?: string }) {
  const node = getArchitectureNode(nodeId);
  const index = architectureNodes.findIndex((item) => item.id === node.id);
  const previous = architectureNodes[(index - 1 + architectureNodes.length) % architectureNodes.length];
  const next = architectureNodes[(index + 1) % architectureNodes.length];
  const prefix = lang === "zh" ? "/zh" : "";
  const otherPrefix = lang === "zh" ? "" : "/zh";
  const l = labels[lang];
  const detailHref = (id: string) => `${prefix}/architecture/detail?node=${id}`;

  return <><SiteHeader lang={lang}/><main className="detail-page">
    <section className="detail-hero shell">
      <div className="detail-breadcrumb"><Link href={`${prefix}/architecture`}>← {l.back}</Link><Link href={`${otherPrefix}/architecture/detail?node=${node.id}`}>{l.switch}</Link></div>
      <div className="detail-title-grid"><span>{node.number}</span><div><div className="eyebrow">{l.layer} / {node.subtitle}</div><h1>{node.title[lang]}</h1><p>{node.summary[lang]}</p></div></div>
    </section>

    <section className="detail-flow shell" aria-label={`${l.input} ${l.process} ${l.output}`}>
      <article><span>{l.input}</span><p>{node.input[lang]}</p></article><i>→</i>
      <article className="detail-process"><span>{l.process}</span><ol>{node.process[lang].map((step) => <li key={step}>{step}</li>)}</ol></article><i>→</i>
      <article><span>{l.output}</span><p>{node.output[lang]}</p></article>
    </section>

    <section className="detail-engineering shell">
      <article className="implementation-card"><span>{l.real}</span><div className="implementation-stack">{node.implementation.map((item, itemIndex) => <div key={item}><b>{String(itemIndex + 1).padStart(2, "0")}</b><code>{item}</code></div>)}</div></article>
      <article className="guard-card"><span>{l.guards}</span><ul>{node.guards[lang].map((guard) => <li key={guard}>{guard}</li>)}</ul></article>
    </section>

    <section className="detail-proof"><div className="shell"><span>{l.proof}</span><p>{node.proof[lang]}</p></div></section>

    <nav className="detail-pager shell" aria-label="Architecture modules">
      <Link href={detailHref(previous.id)}><span>← {l.prev}</span><b>{previous.number} / {previous.title[lang]}</b></Link>
      <Link className="all-modules" href={`${prefix}/architecture`}>{l.map}</Link>
      <Link className="next-module" href={detailHref(next.id)}><span>{l.next} →</span><b>{next.number} / {next.title[lang]}</b></Link>
    </nav>
  </main><SiteFooter lang={lang}/></>;
}
