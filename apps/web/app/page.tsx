import { Workbench } from "./workbench";
import { SiteFooter, SiteHeader } from "./site-shell";

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="hero shell">
          <div className="eyebrow"><span className="pulse" /> VERIFIED AGENT REPLAY</div>
          <h1>See the research agent<br />think in <em>evidence.</em></h1>
          <p className="hero-copy">
            DeepTrace-R1 exposes every plan, search, source decision and citation behind a final answer—so an agent run can be inspected, not merely trusted.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#workbench">Open verified run <span>↘</span></a>
            <a className="button quiet" href="/evaluation">Read evaluation</a>
          </div>
          <div className="metric-strip" aria-label="headline results">
            <div><strong>55.56%</strong><span>Exact Match</span><small>+46.67 pts vs. Base</small></div>
            <div><strong>66.73%</strong><span>Answer F1</span><small>+56.98 pts vs. Base</small></div>
            <div><strong>100%</strong><span>Completion</span><small>90-question held-out set</small></div>
            <div><strong>0</strong><span>Invalid actions</span><small>down from 19</small></div>
          </div>
        </section>

        <section className="manifesto shell">
          <span className="section-index">01 / RUN TRACE</span>
          <div>
            <h2>An answer is only the surface.</h2>
            <p>Inspect the complete execution state: decomposed tasks, tool calls, evidence scores, query rewrites, contradictions, budgets and the final evidence chain.</p>
          </div>
        </section>

        <section className="chain-preview shell">
          <div><span className="section-index">REAL SYSTEM CHAIN</span><h2>Two planes. One evidence trail.</h2><p>The online agent runtime and offline GPU training system are deliberately separated, then joined through versioned adapters and evaluation artifacts.</p><a className="button primary" href="/infrastructure">Inspect infrastructure →</a></div>
          <div className="mini-chain"><span>WEB UI</span><i>→</i><span>LIVE SSE GATEWAY</span><i>→</i><span>SUPERVISOR</span><i>→</i><span>DEEPSEEK + BM25</span><b>SSH CONTROL</b><i>→</i><span>2 × RTX 4090</span><i>→</i><span>QWEN3-8B LORA</span><i>→</i><span>HELD-OUT EVAL</span></div>
        </section>

        <Workbench />

        <section className="proof shell">
          <div className="proof-heading">
            <span className="section-index">02 / SYSTEM PROOF</span>
            <h2>Built to show the work.</h2>
          </div>
          <div className="proof-grid">
            <article><span>LIVE POLICY</span><h3>DeepSeek + typed actions</h3><p>DeepSeek proposes one structured action at a time; the runtime validates tools, budgets and evidence before execution.</p><a href="/architecture">Runtime architecture →</a></article>
            <article><span>RUNTIME</span><h3>Checkpointed execution</h3><p>Every state transition is recorded. Runs can stop, resume and report budget use without losing their evidence ledger.</p><a href="/architecture">System architecture →</a></article>
            <article><span>EVALUATION</span><h3>Held-out, auditable</h3><p>90 questions across HotpotQA, 2WikiMultiHopQA and MuSiQue, with Base and SFT evaluated under the same harness.</p><a href="/evaluation">Evaluation details →</a></article>
          </div>
        </section>

        <section className="truth shell">
          <div><span className="section-index">03 / CLAIM BOUNDARY</span><h2>What this result means.</h2></div>
          <div className="truth-copy">
            <p><strong>Demonstrated:</strong> controlled multi-hop evidence retrieval, trajectory learning, traceable execution and citation-grounded reporting.</p>
            <p><strong>Not claimed:</strong> production open-web reliability. The device-locked streaming backend is live against a pinned BM25 corpus; production Brave Search and open-web drift handling remain future work.</p>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
