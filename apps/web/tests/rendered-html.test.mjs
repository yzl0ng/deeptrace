import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${path}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
}

test("server-renders the independent DeepTrace workbench", async () => {
  const response = await render(); const html = await response.text();
  assert.equal(response.status, 200); assert.match(html, /DeepTrace-R1/); assert.match(html, /VERIFIED AGENT REPLAY/); assert.match(html, /55\.56%/); assert.match(html, /SAVED REAL RUN/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("server-renders the primary evidence pages", async () => {
  for (const [path, phrase] of [["/evaluation", "Competitive signal, with the comparison boundary visible"], ["/architecture", "one navigable map"], ["/training", "Teach the policy from complete runs"]]) {
    const response = await render(path); assert.equal(response.status, 200); assert.match(await response.text(), new RegExp(phrase, "i"));
  }
});

test("evaluation explains datasets, protocol and public method context", async () => {
  const [english, chinese] = await Promise.all([render("/evaluation"), render("/zh/evaluation")]);
  const en = await english.text(); const zh = await chinese.text();
  for (const phrase of ["HotpotQA", "2WikiMultiHopQA", "MuSiQue", "Search-R1 PPO", "IRCoT reproduction", "SPARKLE", "HiGraAgent", "33.70", "56.67", "controlled-evidence protocol"]) assert.ok(en.includes(phrase), `missing evaluation phrase: ${phrase}`);
  for (const phrase of ["用什么数据测试", "公开方法横向参照", "Search-R1 PPO", "IRCoT 统一复现", "以前的人做到什么效果", "受控证据环境"]) assert.ok(zh.includes(phrase), `missing Chinese evaluation phrase: ${phrase}`);
  assert.match(en, /https:\/\/aclanthology\.org\/2026\.findings-eacl\.62/);
  assert.match(zh, /HotpotQA 官方 distractor 榜单/);
});

test("server-renders the complete Chinese experience", async () => {
  for (const [path, phrase] of [["/zh", "看见研究智能体"], ["/zh/infrastructure", "从网页请求到远程 4090 训练"], ["/zh/evaluation", "不只看 Base 提升"], ["/zh/architecture", "一张可以点开的导图"], ["/zh/training", "用完整运行轨迹训练策略"]]) {
    const response = await render(path); assert.equal(response.status, 200); assert.match(await response.text(), new RegExp(phrase));
  }
});

test("shows sanitized real runtime and GPU evidence", async () => {
  const [home, infrastructure] = await Promise.all([render("/"), render("/infrastructure")]); const homeHtml = await home.text(); const infraHtml = await infrastructure.text();
  assert.match(homeHtml, /VERIFIED REAL RUN/); assert.match(homeHtml, /DEEPSEEK-V4-FLASH/); assert.match(infraHtml, /8 × RTX 4090/); assert.match(infraHtml, /GPU 4 \+ GPU 5/); assert.match(infraHtml, /SSH control channel/); assert.doesNotMatch(infraHtml, /PRIVATE KEY|BEGIN OPENSSH|ssh-rsa/i);
});

test("architecture map exposes clickable engineering modules", async () => {
  const [english, chinese] = await Promise.all([render("/architecture"), render("/zh/architecture")]); const en = await english.text(); const zh = await chinese.text();
  for (const href of ["request-api", "supervisor", "policy", "tools", "evidence", "gate-repair", "memory-report", "persistence-recovery", "offline-training"]) {
    assert.match(en, new RegExp(`/architecture/detail\\?node=${href}`));
    assert.match(zh, new RegExp(`/zh/architecture/detail\\?node=${href}`));
  }
  assert.match(en, /RESEARCHRUN \/ 12 STATES/); assert.match(en, /3 parallel searches/);
  assert.match(zh, /点击任意模块查看细节/); assert.match(zh, /模型只提出强类型决策/);
});

test("architecture detail pages explain each block without tables", async () => {
  const [supervisor, training, evidenceZh] = await Promise.all([
    render("/architecture/detail?node=supervisor"),
    render("/architecture/detail?node=offline-training"),
    render("/zh/architecture/detail?node=evidence"),
  ]);
  const supervisorHtml = await supervisor.text(); const trainingHtml = await training.text(); const evidenceZhHtml = await evidenceZh.text();
  assert.equal(supervisor.status, 200); assert.equal(training.status, 200); assert.equal(evidenceZh.status, 200);
  for (const phrase of ["SupervisorResearchService", "ThreadPoolExecutor", "RunStatus (12 values)", "checkpoint v14"]) assert.ok(supervisorHtml.includes(phrase));
  for (const phrase of ["500 → 479", "431 train / 48 validation", "GPU 4 + 5", "Qwen3-8B LoRA adapter"]) assert.ok(trainingHtml.includes(phrase));
  for (const phrase of ["证据账本与溯源", "EvidenceStore", "Source → Document → Passage → Evidence"]) assert.ok(evidenceZhHtml.includes(phrase));
  assert.doesNotMatch(supervisorHtml, /<table/i);
});

test("live runtime streams real state, evidence and report events", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("live-test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const counts = new Map(); const savedRuns = [];
  const database = {
    prepare(sql) {
      let values = [];
      return {
        bind(...next) { values = next; return this; },
        async run() {
          if (sql.includes("live_rate_limits")) { const bucket = String(values[0]); const count = (counts.get(bucket) ?? 0) + 1; counts.set(bucket, count); return { success: true, results: [{ count }] }; }
          if (sql.startsWith("INSERT INTO live_runs")) { savedRuns.push(values); return { success: true, results: [] }; }
          return { success: true, results: [] };
        },
      };
    },
    async batch(statements) { return Promise.all(statements.map((statement) => statement.run())); },
  };
  const originalFetch = globalThis.fetch; let modelCall = 0;
  const policyActions = [
    { rationale_summary: "Search for lexical and semantic retrieval evidence.", action: "search", arguments: { query: "BM25 Dense Retrieval RRF" }, evidence_ids: [], final_answer: null },
    { rationale_summary: "Read the RRF passage returned by search.", action: "read_page", arguments: { evidence_id: "doc-003" }, evidence_ids: ["doc-003"], final_answer: null },
    { rationale_summary: "Read the Dense Retrieval passage returned by search.", action: "read_page", arguments: { evidence_id: "doc-002" }, evidence_ids: ["doc-002"], final_answer: null },
    { rationale_summary: "Evaluate the two read evidence passages.", action: "evaluate_evidence", arguments: {}, evidence_ids: ["doc-002", "doc-003"], final_answer: null },
    { rationale_summary: "The read evidence is sufficient for a grounded answer.", action: "answer", arguments: {}, evidence_ids: ["doc-002", "doc-003"], final_answer: "BM25 and Dense Retrieval are complementary." },
  ];
  globalThis.fetch = async () => {
    const content = modelCall === 0
      ? JSON.stringify({ route: "direct_answer", category: "knowledge_explanation", confidence: 0.99, reason: "This is stable technical knowledge.", final_answer: "An unsupported model-memory answer." })
      : modelCall <= policyActions.length
        ? JSON.stringify(policyActions[modelCall - 1])
        : JSON.stringify({ final_report: "Dense retrieval supplies semantic matching [doc-002], while RRF combines ranked lists without adding incompatible raw scores [doc-003].", cited_evidence_ids: ["doc-002", "doc-003"] });
    modelCall += 1;
    return Response.json({ model: "deepseek-v4-flash", choices: [{ message: { content } }], usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } });
  };
  try {
    const pending = [];
    const response = await worker.fetch(new Request("http://localhost/api/live-research/runs", { method: "POST", headers: { "content-type": "application/json", cookie: "live_demo_access=test-access" }, body: JSON.stringify({ query: "Why do BM25, Dense Retrieval and RRF work well together?" }) }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) }, DB: database, DEEPSEEK_API_KEY: "test-key", DEEPSEEK_MODEL: "deepseek-v4-flash", LIVE_DEMO_ACCESS_TOKEN: "test-access" }, { waitUntil(promise) { pending.push(promise); }, passThroughOnException() {} });
    const stream = await response.text(); await Promise.all(pending);
    assert.equal(response.status, 200); assert.match(response.headers.get("content-type"), /text\/event-stream/);
    for (const phrase of ["ResearchRun created", "Runtime upgraded route to evidence research", '"model_route":"direct_answer"', '"runtime_override":true', "DeepSeek action policy connected", '"actor":"deepseek"', '"action":"local_search"', '"action":"read_page"', '"action":"evaluate_evidence"', '"contract":"evidence_grade"', '"usage_by_model"', "Evidence gate passed", '"status":"completed"', "doc-002"]) assert.ok(stream.includes(phrase), `missing streamed phrase: ${phrase}; tail=${stream.slice(-1200)}`);
    assert.equal(modelCall, 7); assert.equal(savedRuns.length, 1); assert.equal((stream.match(/\"type\":\"(?:completed|error)\"/g) || []).length, 1);
  } finally { globalThis.fetch = originalFetch; }
});

test("direct-answer gate lets DeepSeek answer a simple question without search", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("direct-test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const savedRuns = [];
  const database = { prepare(sql) { let values = []; return { bind(...next) { values = next; return this; }, async run() { if (sql.startsWith("INSERT INTO live_runs")) savedRuns.push(values); return { success: true, results: [] }; } }; } };
  const originalFetch = globalThis.fetch;
  let modelCalls = 0;
  const requestBodies = [];
  globalThis.fetch = async (_url, init) => {
    modelCalls += 1; requestBodies.push(JSON.parse(String(init.body)));
    if (modelCalls === 1) return Response.json({ model: "deepseek-v4-flash", choices: [{ finish_reason: "length", message: { content: "", reasoning_content: "internal reasoning is intentionally not consumed" } }], usage: { prompt_tokens: 20, completion_tokens: 5, total_tokens: 25 } });
    const content = JSON.stringify({ route: "direct_answer", category: "arithmetic", confidence: 0.99, reason: "Simple arithmetic does not require external evidence.", final_answer: "10 + 10 = 20。" });
    return Response.json({ model: "deepseek-v4-flash", choices: [{ finish_reason: "stop", message: { content } }], usage: { prompt_tokens: 40, completion_tokens: 20, total_tokens: 60 } });
  };
  try {
    const pending = [];
    const response = await worker.fetch(new Request("http://localhost/api/live-research/runs", { method: "POST", headers: { "content-type": "application/json", cookie: "live_demo_access=test-access" }, body: JSON.stringify({ query: "10加10等于几" }) }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) }, DB: database, DEEPSEEK_API_KEY: "test-key", LIVE_DEMO_ACCESS_TOKEN: "test-access" }, { waitUntil(promise) { pending.push(promise); }, passThroughOnException() {} });
    const stream = await response.text(); await Promise.all(pending);
    assert.equal(response.status, 200); assert.equal(modelCalls, 2); assert.equal(savedRuns.length, 1); assert.equal((stream.match(/\"type\":\"(?:completed|error)\"/g) || []).length, 1);
    assert.ok(requestBodies.every((body) => body.thinking?.type === "disabled")); assert.ok(requestBodies.every((body) => body.max_tokens >= 1200)); assert.match(requestBodies[1].messages.at(-1).content, /IMPORTANT RETRY/);
    for (const phrase of ["Runtime confirmed direct answer", '"route":"direct_answer"', '"category":"arithmetic"', '"runtime_override":false', '"search_required":false', '"search_count":0', "10 + 10 = 20", '"status":"completed"', '"total_tokens":85']) assert.ok(stream.includes(phrase), `missing direct-route phrase: ${phrase}`);
    assert.doesNotMatch(stream, /local_search|DeepSeek action policy connected|Evidence gate rejected/);
  } finally { globalThis.fetch = originalFetch; }
});

test("live workbench lets users inspect every streamed step return value", async () => {
  const source = await readFile(new URL("../app/workbench.tsx", import.meta.url), "utf8");
  for (const phrase of ["setSelectedLiveSeq(event.seq)", "aria-pressed", "步骤返回结果", "真实返回结果 RETURNED OUTPUT", "原始 SSE 事件 JSON", "createResearchRun()", "executePolicyAction()", "当前实时链路只依赖 DeepSeek"]) assert.ok(source.includes(phrase));
});

test("live runtime requires a device unlock cookie without limiting runs", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url); workerUrl.searchParams.set("unlock-test", `${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) }, DB: {}, DEEPSEEK_API_KEY: "test-key", LIVE_DEMO_ACCESS_TOKEN: "private-demo-code" };
  const status = await worker.fetch(new Request("http://localhost/api/live-research/status"), env, { waitUntil() {}, passThroughOnException() {} });
  const statusPayload = await status.json(); assert.equal(statusPayload.ready, true); assert.equal(statusPayload.authorized, false); assert.match(statusPayload.limits.runs, /unlimited/i);
  const unlock = await worker.fetch(new Request("http://localhost/api/live-research/unlock", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ token: "private-demo-code" }) }), env, { waitUntil() {}, passThroughOnException() {} });
  assert.equal(unlock.status, 200); assert.match(unlock.headers.get("set-cookie"), /live_demo_access=.*HttpOnly.*Max-Age=2592000/);
});
