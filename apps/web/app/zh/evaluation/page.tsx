import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "../../site-shell";

export const metadata: Metadata = { title: "评估结果" };

const datasets = [
  { name: "HotpotQA", count: "独立 30 题", description: "基于 Wikipedia 的跨文档桥接与比较问题，并提供句子级支持事实，考察能否把两份证据连起来。", em: "66.67%", f1: "76.56%", recall: "55.00%", href: "https://aclanthology.org/D18-1259/" },
  { name: "2WikiMultiHopQA", count: "独立 30 题", description: "同时利用结构化与非结构化 Wikipedia 信息构造多跳问题，并提供显式推理路径。", em: "53.33%", f1: "62.92%", recall: "60.00%", href: "https://aclanthology.org/2020.coling-main.580/" },
  { name: "MuSiQue", count: "独立 30 题", description: "由相互依赖的单跳问题组合成 2–4 跳问题，专门减少模型靠单条线索走捷径的可能。", em: "46.67%", f1: "60.72%", recall: "56.67%", href: "https://aclanthology.org/2022.tacl-1.31/" },
];

const emMethods = [
  { name: "DeepTrace-R1", setting: "Qwen3-8B · 500 条候选 / 479 条通过 · 无 RL", hotpot: "66.67", wiki: "53.33", musique: "46.67", macro: "55.56", width: "98.0%", ours: true },
  { name: "Search-R1 PPO", setting: "Qwen2.5-7B · E5 · 2018 Wikipedia", hotpot: "43.30", wiki: "38.20", musique: "19.60", macro: "33.70", width: "59.5%" },
  { name: "IRCoT 统一复现", setting: "GPT-4o-mini · 迭代式非结构化检索", hotpot: "50.10", wiki: "62.60", musique: "16.70", macro: "43.13", width: "76.1%" },
  { name: "HiGraAgent", setting: "GPT-4o-mini · 层次图 · 双 Agent", hotpot: "57.40", wiki: "73.60", musique: "39.00", macro: "56.67", width: "100%" },
];

export default function Page() {
  return <><SiteHeader lang="zh"/><main>
    <section className="page-hero shell"><div className="eyebrow">评估 / 独立测试 + 公开方法参照</div><h1>不只看 Base 提升，也看它在公开方法中的位置。</h1><p>DeepTrace-R1 在 90 道未见多跳问题上达到 55.56% Macro EM、66.73% Macro F1。这里把“同协议严格对比”和“不同论文的方向性横向参照”分开呈现。</p></section>

    <section className="content-section shell"><div className="section-heading"><span className="section-index">01 / 用什么数据测试</span><h2>三个数据集，覆盖三类证据链推理。</h2></div>
      <div className="dataset-grid">{datasets.map((dataset) => <article key={dataset.name}><div><span>{dataset.count}</span><a href={dataset.href} target="_blank" rel="noreferrer">官方论文 ↗</a></div><h3>{dataset.name}</h3><p>{dataset.description}</p><dl><div><dt>EM</dt><dd>{dataset.em}</dd></div><div><dt>F1</dt><dd>{dataset.f1}</dd></div><div><dt>证据召回</dt><dd>{dataset.recall}</dd></div></dl></article>)}</div>
      <div className="eval-protocol" aria-label="评估协议"><article><b>01</b><span>冻结样本</span><p>从三个官方 validation split 各取 30 题；与 500 条训练 seed、47 条 dev、6 条早期测试的重合均为 0。</p></article><article><b>02</b><span>提供证据</span><p>每题只暴露冻结的 Gold 支持段落；没有 distractor 语料、FullWiki 索引，也不是实时 Web 搜索。</p></article><article><b>03</b><span>同框运行</span><p>Base 与 SFT 使用完全相同的 Runtime、动作预算、工具、证据门控和最终答案协议。</p></article><article><b>04</b><span>统一计分</span><p>统计标准化答案 EM/F1、Gold 证据召回、完成率、非法动作和最终协议失败。</p></article></div>
    </section>

    <section className="content-section shell"><div className="section-heading"><span className="section-index">02 / 严格内部对比</span><h2>完全相同评估框架下的 Base → SFT。</h2></div>
      <div className="data-table"><div className="data-row header"><span>指标</span><span>Base</span><span>SFT</span><span>变化</span></div><div className="data-row"><span>精确匹配率 EM</span><strong>8.89%</strong><strong className="win">55.56%</strong><span>+46.67 点</span></div><div className="data-row"><span>答案 F1</span><strong>9.75%</strong><strong className="win">66.73%</strong><span>+56.98 点</span></div><div className="data-row"><span>任务完成率</span><strong>15.56%</strong><strong className="win">100%</strong><span>+84.44 点</span></div><div className="data-row"><span>证据召回率</span><strong>25.28%</strong><strong className="win">57.22%</strong><span>+31.94 点</span></div></div>
      <div className="big-number-grid reliability-grid"><div><strong>19 → 0</strong><span>非法动作</span><p>格式错误或 Runtime 不支持的动作被消除。</p></div><div><strong>65 → 0</strong><span>最终协议失败</span><p>缺少规定终止答案结构的运行被消除。</p></div><div><strong>90 / 90</strong><span>成功完成</span><p>所有 SFT 运行都进入有效终态。</p></div><div><strong>2 轮</strong><span>训练投入</span><p>431 条训练 / 48 条验证轨迹；LoRA，无 RL。</p></div></div>
    </section>

    <section className="content-section shell comparison-section"><div className="section-heading"><span className="section-index">03 / 公开方法横向参照</span><h2>放到搜索 Agent 研究里，大概处在什么位置。</h2></div>
      <div className="position-grid"><article><strong>+21.86</strong><span>相对 Search-R1 PPO 的 Macro EM 点数</span><p>Search-R1 是同为约 7B 规模的 RL 搜索策略参照；但我们的受控环境更容易，所以这是位置参考，不是严格胜负。</p></article><article><strong>−1.11</strong><span>相对 HiGraAgent 的 Macro EM 点数</span><p>只用 479 条通过筛选的 SFT 轨迹、没有 RL，Macro EM 已接近采用 GPT-4o-mini、层次知识图谱和双 Agent 的系统。</p></article></div>
      <div className="macro-chart" aria-label="Macro EM 对比">{emMethods.map((method) => <div className={method.ours ? "ours" : ""} key={method.name}><span>{method.name}</span><i><b style={{width: method.width}}/></i><strong>{method.macro}%</strong></div>)}</div>
      <div className="comparison-table-wrap"><div className="comparison-table"><div className="comparison-row header"><span>系统 / 设置</span><span>HotpotQA</span><span>2Wiki</span><span>MuSiQue</span><span>Macro EM</span></div>{emMethods.map((method) => <div className={`comparison-row${method.ours ? " ours" : ""}`} key={method.name}><span><strong>{method.name}</strong><small>{method.setting}</small></span><span>{method.hotpot}%</span><span>{method.wiki}%</span><span>{method.musique}%</span><span><strong>{method.macro}%</strong></span></div>)}</div></div>
      <h3 className="subsection-title">答案 F1 参照</h3>
      <div className="comparison-table-wrap"><div className="comparison-table"><div className="comparison-row header"><span>系统 / 设置</span><span>HotpotQA</span><span>2Wiki</span><span>MuSiQue</span><span>Macro F1</span></div><div className="comparison-row ours"><span><strong>DeepTrace-R1</strong><small>Qwen3-8B · 受控环境</small></span><span>76.56%</span><span>62.92%</span><span>60.72%</span><span><strong>66.73%</strong></span></div><div className="comparison-row"><span><strong>SPARKLE</strong><small>Qwen2.5-7B · 自适应 Agentic RAG</small></span><span>63.14%</span><span>64.78%</span><span>32.85%</span><span><strong>53.59%</strong></span></div><div className="comparison-row"><span><strong>HiGraAgent</strong><small>GPT-4o-mini · 层次图 + 双 Agent</small></span><span>74.80%</span><span>80.40%</span><span>58.00%</span><span><strong>71.07%</strong></span></div></div></div>
      <div className="note"><strong>怎么理解这张表：</strong>公开方法数字来自各自论文或文中注明的统一复现，不是在我们的代码、90 题样本和受控语料上重新跑出的。检索库、模型、样本量与预算均不同，因此只能说明大致研究位置；上面的 Base→SFT 才是严格因果对比。</div>
    </section>

    <section className="content-section shell"><div className="section-heading"><span className="section-index">04 / 以前的人做到什么效果</span><h2>用官方榜单给出一个专业系统上限参照。</h2></div>
      <div className="upper-reference"><div><span>HotpotQA 官方 distractor 榜单</span><strong>72.69 EM / 85.04 F1</strong><p>Beam Retrieval 单模型成绩。它是专门训练的完整榜单系统，与我们的样本、模型和评估预算都不同。</p></div><div><span>DeepTrace-R1 · 受控 HotpotQA</span><strong>66.67 EM / 76.56 F1</strong><p>距离专业系统参照约 6.02 EM、8.48 F1 点；这个差距可以参考，但不能把不同设置包装成同一榜单排名。</p></div></div>
      <div className="source-links"><span>一手资料</span><a href="https://arxiv.org/abs/2503.09516" target="_blank" rel="noreferrer">Search-R1 ↗</a><a href="https://aclanthology.org/2023.acl-long.557/" target="_blank" rel="noreferrer">IRCoT ↗</a><a href="https://aclanthology.org/2026.acl-long.1793/" target="_blank" rel="noreferrer">SPARKLE ↗</a><a href="https://aclanthology.org/2026.findings-eacl.62/" target="_blank" rel="noreferrer">HiGraAgent ↗</a><a href="https://hotpotqa.github.io/" target="_blank" rel="noreferrer">HotpotQA 榜单 ↗</a></div>
    </section>

    <section className="truth shell"><div><span className="section-index">05 / 结论边界</span><h2>原型效果很强，但还不能宣称开放检索 SOTA。</h2></div><div className="truth-copy"><p><strong>可以证明：</strong>过滤后的完整轨迹让 Qwen3-8B 学会了更可靠的工具协议，并在冻结的受控证据环境中显著提升未见问题的答案质量。</p><p><strong>尚未证明：</strong>在 distractor、FullWiki 或实时 Web Search 下超过公开系统。下一步应在冻结的生产式检索语料上扩大样本，并通过多次重复实验计算置信区间。</p></div></section>
  </main><SiteFooter lang="zh"/></>;
}
