"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

const navLabels = { en: ["Workbench", "System", "Evaluation", "Architecture", "Training"], zh: ["工作台", "真实链路", "评估结果", "系统架构", "训练流程"] };
const routeNames = ["", "infrastructure", "evaluation", "architecture", "training"];

export function SiteHeader({ lang = "en" }: { lang?: "en" | "zh" }) {
  const path = usePathname();
  const prefix = lang === "zh" ? "/zh" : "";
  const links = routeNames.map((route, index) => [route ? `${prefix}/${route}` : (prefix || "/"), navLabels[lang][index]]);
  const route = path.replace(/^\/zh/, "");
  const switchHref = lang === "zh" ? (route || "/") : `/zh${route === "/" ? "" : route}`;
  return <header className="site-header"><div className="shell nav-wrap">
    <Link className="brand" href="/"><span className="brand-mark">D/1</span><span>DeepTrace—R1</span></Link>
    <nav aria-label="Primary navigation">{links.map(([href,label]) => <Link key={href} className={path === href ? "active" : ""} href={href}>{label}</Link>)}</nav>
    <div className="nav-status"><span className="status-copy"><span className="status-dot" />{lang === "zh" ? "研究版本 · 0.3" : "research build · 0.3"}</span><Link className="lang-switch" href={switchHref}>{lang === "zh" ? "EN" : "中文"}</Link></div>
  </div></header>;
}

export function SiteFooter({ lang = "en" }: { lang?: "en" | "zh" }) {
  return <footer><div className="shell footer-grid"><div><Link className="brand" href={lang === "zh" ? "/zh" : "/"}><span className="brand-mark">D/1</span><span>DeepTrace—R1</span></Link><p>{lang === "zh" ? "可审计的研究型智能体执行系统。" : "Inspectable research-agent execution."}</p></div><div><span>{lang === "zh" ? "实验结果" : "RESULTS"}</span><p>{lang === "zh" ? <>90 道独立测试题<br/>14 / 14 端到端验收通过</> : <>90 held-out questions<br/>14 / 14 E2E acceptance</>}</p></div><div><span>{lang === "zh" ? "当前状态" : "STATUS"}</span><p>{lang === "zh" ? <>研究原型<br/>受控证据环境</> : <>Research prototype<br/>Controlled evidence environment</>}</p></div></div></footer>;
}
