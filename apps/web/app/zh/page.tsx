import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "../site-shell";
import { Workbench } from "../workbench";
export const metadata: Metadata = { title: "DeepTrace-R1 — 可审计研究型智能体", description: "展示规划、检索、证据评估与最终报告全过程的可审计研究型智能体系统。" };

export default function ChineseHome(){return <><SiteHeader lang="zh"/><main>
  <section className="hero shell"><div className="eyebrow"><span className="pulse"/> 已验证智能体轨迹回放</div><h1>看见研究智能体<br/>如何用<em>证据思考。</em></h1><p className="hero-copy">DeepTrace-R1 将规划、检索、来源判断和引用依据完整呈现——让一次智能体运行能够被检查，而不只是被相信。</p><div className="hero-actions"><a className="button primary" href="#workbench">打开已验证轨迹 <span>↘</span></a><a className="button quiet" href="/zh/evaluation">查看评估结果</a></div>
  <div className="metric-strip"><div><strong>55.56%</strong><span>精确匹配率</span><small>相比 Base +46.67 点</small></div><div><strong>66.73%</strong><span>答案 F1</span><small>相比 Base +56.98 点</small></div><div><strong>100%</strong><span>任务完成率</span><small>90 道独立测试题</small></div><div><strong>0</strong><span>无效动作</span><small>由 19 次降至 0</small></div></div></section>
  <section className="manifesto shell"><span className="section-index">01 / 运行轨迹</span><div><h2>答案只是系统最表层的输出。</h2><p>深入检查完整执行状态：任务拆解、工具调用、证据评分、查询重写、矛盾记录、预算消耗以及最终证据链。</p></div></section>
  <section className="chain-preview shell"><div><span className="section-index">真实系统链路</span><h2>两条执行平面，一条证据链。</h2><p>在线智能体运行时与离线 GPU 训练系统相互隔离，再通过版本化 Adapter 和评估产物完成闭环。</p><a className="button primary" href="/zh/infrastructure">查看真实基础设施 →</a></div><div className="mini-chain"><span>网页界面</span><i>→</i><span>实时 SSE 网关</span><i>→</i><span>SUPERVISOR</span><i>→</i><span>DEEPSEEK + BM25</span><b>SSH 控制</b><i>→</i><span>2 × RTX 4090</span><i>→</i><span>QWEN3-8B LORA</span><i>→</i><span>独立评估</span></div></section>
  <Workbench lang="zh"/>
  <section className="proof shell"><div className="proof-heading"><span className="section-index">02 / 系统证据</span><h2>不是只给答案，而是展示过程。</h2></div><div className="proof-grid">
    <article><span>在线策略</span><h3>DeepSeek + 强类型动作</h3><p>DeepSeek 每次只提议一个结构化动作；Runtime 校验工具、预算与证据后才会执行。</p><a href="/zh/architecture">查看运行时架构 →</a></article><article><span>执行运行时</span><h3>支持检查点的执行系统</h3><p>记录每一次状态转换；任务可停止、恢复并报告预算使用，同时保持证据账本完整。</p><a href="/zh/architecture">查看系统架构 →</a></article><article><span>独立评估</span><h3>留出测试，可追溯</h3><p>在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 各 30 道问题上，用同一套评估框架比较 Base 与 SFT。</p><a href="/zh/evaluation">查看评估详情 →</a></article>
  </div></section>
  <section className="truth shell"><div><span className="section-index">03 / 结论边界</span><h2>这些结果究竟说明什么？</h2></div><div className="truth-copy"><p><strong>已经证明：</strong>受控多跳证据检索、轨迹学习、可追踪执行和基于引用的报告生成。</p><p><strong>尚未宣称：</strong>生产级开放网络可靠性。本机解锁的流式后端已经基于固定 BM25 语料运行；生产 Brave Search 和网页漂移处理仍属于后续工作。</p></div></section>
</main><SiteFooter lang="zh"/></>}
