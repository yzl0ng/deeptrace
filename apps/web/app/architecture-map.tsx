import Link from "next/link";
import { architectureNodes, type ArchitectureLanguage } from "./architecture-data";
import { SiteFooter, SiteHeader } from "./site-shell";

const copy = {
  en: {
    eyebrow: "INTERACTIVE SYSTEM MAP / CLICK TO EXPLAIN",
    title: "Explain the real agent system as one navigable map.",
    lead: "Start at the question, follow the solid arrows to the cited report, then use the side loops to explain evidence repair, recovery and offline training. Every block opens its engineering detail page.",
    hint: "CLICK ANY BLOCK",
    main: "ONLINE RESEARCH RUNTIME",
    evidence: "EVIDENCE CONTROL LOOP",
    reliability: "DURABILITY SIDE PATH",
    training: "OFFLINE POLICY IMPROVEMENT",
    click: "OPEN DETAIL",
    stateTitle: "The runtime state machine inside block 02",
    stateCopy: "The happy path is linear to explain; four explicit branches make failure and human clarification visible.",
    branches: ["scoping → awaiting_clarification", "active → cancelled", "budget check → budget_exceeded", "exception → failed"],
    boundaryTitle: "The interview sentence to remember",
    boundary: "The model proposes typed decisions; the Supervisor owns execution; the evidence ledger decides what may be cited.",
    runLabel: "SAVED REAL RUN",
    run: "3 parallel searches · 3/3 evidence sufficient · 9 model steps · 9,919 tokens · checkpoint v14 · 14/14 acceptance",
  },
  zh: {
    eyebrow: "可交互系统导图 / 点击即可讲解",
    title: "用一张可以点开的导图，讲清真实 Agent 系统。",
    lead: "从用户问题开始，沿实线箭头讲到带引用的报告；再通过侧向回路说明证据修复、断点恢复和离线训练。每个模块都能点进独立工程详情页。",
    hint: "点击任意模块查看细节",
    main: "在线研究主链路",
    evidence: "证据控制回路",
    reliability: "持久化与恢复旁路",
    training: "离线策略改进链路",
    click: "打开详情",
    stateTitle: "02 号模块内部的状态机",
    stateCopy: "成功主路径保持线性，四条显式分支让澄清、取消、超预算和失败都可见。",
    branches: ["scoping → awaiting_clarification", "活动状态 → cancelled", "预算检查 → budget_exceeded", "异常 → failed"],
    boundaryTitle: "面试时最值得记住的一句话",
    boundary: "模型只提出强类型决策，Supervisor 掌握执行权，证据账本决定哪些内容可以被引用。",
    runLabel: "真实保存运行",
    run: "3 路并行检索 · 3/3 证据充分 · 9 个模型步骤 · 9,919 tokens · checkpoint v14 · 14/14 验收",
  },
};

function NodeCard({ id, lang }: { id: string; lang: ArchitectureLanguage }) {
  const node = architectureNodes.find((item) => item.id === id)!;
  const prefix = lang === "zh" ? "/zh" : "";
  return <Link className={`map-node lane-${node.lane}`} href={`${prefix}/architecture/detail?node=${node.id}`}>
    <span className="map-node-number">{node.number}</span>
    <span className="map-node-lane">{node.subtitle}</span>
    <h3>{node.title[lang]}</h3>
    <p>{node.summary[lang]}</p>
    <b>{copy[lang].click} <i>↗</i></b>
  </Link>;
}

export function ArchitectureMap({ lang }: { lang: ArchitectureLanguage }) {
  const c = copy[lang];
  return <><SiteHeader lang={lang}/><main className="architecture-map-page">
    <section className="page-hero shell map-hero">
      <div className="eyebrow">{c.eyebrow}</div><h1>{c.title}</h1><p>{c.lead}</p>
      <div className="map-click-hint"><span>↘</span>{c.hint}</div>
    </section>

    <section className="system-map shell" aria-label={c.main}>
      <div className="map-lane-label"><span>01</span>{c.main}</div>
      <div className="runtime-map-row">
        <NodeCard id="request-api" lang={lang}/><span className="map-arrow">→</span>
        <NodeCard id="supervisor" lang={lang}/><span className="map-arrow">→</span>
        <NodeCard id="policy" lang={lang}/><span className="map-arrow">→</span>
        <NodeCard id="memory-report" lang={lang}/>
      </div>

      <div className="map-down-arrow"><span>│</span><b>tool command</b><span>↓</span></div>
      <div className="map-lane-label secondary"><span>02</span>{c.evidence}</div>
      <div className="evidence-map-row">
        <NodeCard id="tools" lang={lang}/><span className="map-arrow">→</span>
        <NodeCard id="evidence" lang={lang}/><span className="map-arrow">→</span>
        <NodeCard id="gate-repair" lang={lang}/>
      </div>
      <div className="repair-loop"><span>└──── insufficient evidence: rewrite → search → re-grade ────┘</span><b>accepted evidence ↑</b></div>

      <div className="side-path-grid">
        <div><div className="map-lane-label secondary"><span>03</span>{c.reliability}</div><NodeCard id="persistence-recovery" lang={lang}/></div>
        <div className="side-link"><span>← checkpoint every durable step →</span></div>
        <div><div className="map-lane-label secondary"><span>04</span>{c.training}</div><NodeCard id="offline-training" lang={lang}/></div>
      </div>
      <div className="training-return"><span>Teacher traces → validated dataset → 2×4090 → Qwen LoRA</span><b>{lang === "zh" ? "历史训练实验；当前演示不依赖它" : "historical training experiment; not required by the current demo"}</b></div>
    </section>

    <section className="content-section dark-section"><div className="shell state-map-panel">
      <div><span className="section-index">RESEARCHRUN / 12 STATES</span><h2>{c.stateTitle}</h2><p>{c.stateCopy}</p></div>
      <div className="compact-state-flow"><span>queued</span><i>→</i><span>scoping</span><i>→</i><span>planning</span><i>→</i><span>researching</span><i>→</i><span>checking_evidence</span><i>↻</i><span>compressing</span><i>→</i><span>writing</span><i>→</i><span className="success">completed</span></div>
      <div className="compact-branches">{c.branches.map((branch) => <code key={branch}>{branch}</code>)}</div>
    </div></section>

    <section className="map-takeaway shell"><div><span>{c.boundaryTitle}</span><blockquote>“{c.boundary}”</blockquote></div><div className="saved-run"><span>{c.runLabel}</span><p>{c.run}</p></div></section>
  </main><SiteFooter lang={lang}/></>;
}
