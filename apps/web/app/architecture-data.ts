export type ArchitectureLanguage = "en" | "zh";

export type ArchitectureNode = {
  id: string;
  number: string;
  lane: "entry" | "runtime" | "evidence" | "reliability" | "training";
  title: Record<ArchitectureLanguage, string>;
  subtitle: string;
  summary: Record<ArchitectureLanguage, string>;
  input: Record<ArchitectureLanguage, string>;
  process: Record<ArchitectureLanguage, string[]>;
  output: Record<ArchitectureLanguage, string>;
  implementation: string[];
  guards: Record<ArchitectureLanguage, string[]>;
  proof: Record<ArchitectureLanguage, string>;
};

export const architectureNodes: ArchitectureNode[] = [
  {
    id: "request-api", number: "01", lane: "entry", title: { en: "Request & API", zh: "请求与 API 入口" }, subtitle: "Browser → FastAPI",
    summary: { en: "Turns one research question into a persisted ResearchRun and returns a run_id immediately.", zh: "把一个研究问题转换成可持久化的 ResearchRun，并立即返回 run_id。" },
    input: { en: "User query, mode, model and hard budget configuration.", zh: "用户问题、运行模式、模型名与硬预算配置。" },
    process: { en: ["POST /runs validates the request.", "A typed ResearchRun is created in queued state.", "The caller polls GET /runs/{run_id} for trace and result."], zh: ["POST /runs 校验请求。", "创建 queued 状态的强类型 ResearchRun。", "调用方通过 GET /runs/{run_id} 获取轨迹与结果。"] },
    output: { en: "run_id plus a durable aggregate that can be inspected, cancelled or resumed.", zh: "run_id，以及可查看、取消、恢复的持久化聚合对象。" },
    implementation: ["FastAPI run routes", "ResearchRun (Pydantic)", "AgenticRunRepository"],
    guards: { en: ["Reject malformed budgets and unsupported modes.", "Never expose SSH credentials or model secrets."], zh: ["拒绝非法预算和不支持的模式。", "绝不把 SSH 凭据或模型密钥暴露到页面。"] },
    proof: { en: "The saved run can be loaded back from SQLite with its complete trace.", zh: "真实保存的运行可以从 SQLite 完整回读全部轨迹。" },
  },
  {
    id: "supervisor", number: "02", lane: "runtime", title: { en: "Deterministic Supervisor", zh: "确定性 Supervisor 编排" }, subtitle: "SupervisorResearchService",
    summary: { en: "The control plane owns every transition; the model never controls threads, budgets or database writes.", zh: "控制平面负责所有状态迁移；模型不控制线程、预算和数据库写入。" },
    input: { en: "A queued or resumable ResearchRun.", zh: "一个 queued 状态或可恢复的 ResearchRun。" },
    process: { en: ["Advance through 12 persisted statuses.", "Invoke one named model stage at a time.", "Schedule bounded research futures and checkpoint their completion."], zh: ["在 12 个持久化状态之间推进。", "每次只调用一个命名模型阶段。", "调度受限的并行研究任务，并在完成后打检查点。"] },
    output: { en: "A completed, cancelled, failed or budget_exceeded run with an auditable stop reason.", zh: "一个 completed、cancelled、failed 或 budget_exceeded 的运行，并带可审计停止原因。" },
    implementation: ["SupervisorResearchService", "ThreadPoolExecutor", "RunStatus (12 values)", "RunCheckpoint"],
    guards: { en: ["Reserve search quota before futures launch.", "Re-read cancellation at durable boundaries.", "Skip completed stages during resume."], zh: ["Future 启动前预留检索额度。", "在持久化边界重新读取取消标志。", "恢复时跳过 completed_stages。"] },
    proof: { en: "Real run: 9 model steps, 3 searches, checkpoint v14 and zero runtime errors.", zh: "真实运行：9 个模型步骤、3 次检索、checkpoint v14、0 个运行错误。" },
  },
  {
    id: "policy", number: "03", lane: "runtime", title: { en: "Model Policy", zh: "模型策略层" }, subtitle: "ResearchModel.invoke(stage, payload)",
    summary: { en: "A replaceable policy produces typed JSON decisions, not executable code or unrestricted agent chatter.", zh: "可替换策略只产生强类型 JSON 决策，不生成可执行代码，也不进行无约束自由对话。" },
    input: { en: "A named stage and a bounded payload assembled by the Supervisor.", zh: "Supervisor 组装的命名 stage 与受限 payload。" },
    process: { en: ["scope → brief → plan", "evidence_grade ↔ query_rewrite", "contradictions → memory_fold → report"], zh: ["scope → brief → plan", "evidence_grade ↔ query_rewrite", "contradictions → memory_fold → report"] },
    output: { en: "Exactly one JSON object matching one of eight stage contracts.", zh: "严格返回一个符合八种阶段契约之一的 JSON 对象。" },
    implementation: ["ResearchModel protocol", "DeepSeekResearchModel", "ModelOutput", "StructuredModelError"],
    guards: { en: ["Unknown stages are rejected.", "Invalid structured output has bounded retries.", "Tool and web text is marked untrusted."], zh: ["拒绝未知阶段。", "结构化输出失败只进行有限重试。", "工具和网页文本始终标记为不可信。"] },
    proof: { en: "Both the current action-policy call and grounded-writer call use deepseek-v4-flash; the deterministic runtime remains in control.", zh: "当前动作决策与证据写作都调用 deepseek-v4-flash；确定性 Runtime 始终掌握执行权。" },
  },
  {
    id: "tools", number: "04", lane: "evidence", title: { en: "Controlled Tool Execution", zh: "受控工具执行" }, subtitle: "search / read_page",
    summary: { en: "The runtime—not the model—executes configured tools and records every call as an immutable observation.", zh: "由 Runtime 而不是模型执行已配置工具，并把每次调用记录为不可变 observation。" },
    input: { en: "Validated search query or URL/page target.", zh: "校验后的搜索查询或 URL / 页面目标。" },
    process: { en: ["LocalSearchTool runs BM25 on the pinned corpus.", "WebEvidenceTool composes search, safe reading, storage and retrieval.", "Each call receives a stable call-* identifier."], zh: ["LocalSearchTool 在固定评估语料上执行 BM25。", "WebEvidenceTool 组合搜索、安全读页、存储与召回。", "每次调用获得稳定的 call-* 标识。"] },
    output: { en: "ToolCallRecord plus bounded observation and evidence IDs.", zh: "ToolCallRecord、受限 observation 与 evidence ID。" },
    implementation: ["ResearchTool", "LocalSearchTool", "WebEvidenceTool", "BraveSearchProvider", "SafePageReader"],
    guards: { en: ["Block private/reserved IPs and URL credentials.", "Allow HTTP(S) only; enforce MIME, size, redirect and page budgets."], zh: ["拦截私网 / 保留 IP 和 URL 凭据。", "仅允许 HTTP(S)，并限制 MIME、大小、跳转与读页预算。"] },
    proof: { en: "The saved run issued three parallel local_search calls and persisted all three observations.", zh: "真实运行并行发起 3 次 local_search，并持久化全部 observation。" },
  },
  {
    id: "evidence", number: "05", lane: "evidence", title: { en: "Evidence Ledger", zh: "证据账本与溯源" }, subtitle: "Source → Document → Passage → Evidence",
    summary: { en: "Every usable claim is tied to a stable evidence identity and a reproducible source chain.", zh: "每个可用主张都绑定稳定证据身份，以及可复现的来源链。" },
    input: { en: "Search results and safely fetched page content.", zh: "搜索结果与经过安全读取的页面内容。" },
    process: { en: ["Canonicalize source URLs.", "Hash and deduplicate documents.", "Split documents into passages and assign stable evidence IDs."], zh: ["规范化来源 URL。", "对文档做 hash 与去重。", "切分 Passage 并分配稳定 evidence ID。"] },
    output: { en: "Evidence records with provenance, content hash and retrieval metadata.", zh: "带溯源、内容 hash 和召回元数据的 Evidence 记录。" },
    implementation: ["EvidenceStore", "EvidenceRetriever", "Source", "WebDocument", "Passage", "Evidence"],
    guards: { en: ["Web text remains untrusted data.", "Tool-only evidence falls back to tool:{call_id}."], zh: ["网页文本始终是不可信数据。", "没有 Passage ID 的工具证据退化为 tool:{call_id}。"] },
    proof: { en: "Final citations are validated against the evidence IDs produced in this ledger.", zh: "最终引用只能来自这本账本中生成的 evidence ID。" },
  },
  {
    id: "gate-repair", number: "06", lane: "evidence", title: { en: "Evidence Gate & Repair Loop", zh: "证据门控与修复回路" }, subtitle: "grade → rewrite → search → re-grade",
    summary: { en: "Weak evidence cannot silently flow into the answer; it is graded, repaired within budget or reported as insufficient.", zh: "弱证据不能静默进入答案；系统会评分、在预算内修复，或明确报告证据不足。" },
    input: { en: "Subtask observations, attempt number and evidence IDs.", zh: "子任务 observation、attempt 次数与 evidence ID。" },
    process: { en: ["evidence_grade returns sufficient, reason and missing questions.", "If insufficient, query_rewrite produces a focused query.", "Re-search and re-grade; then run contradiction detection."], zh: ["evidence_grade 返回 sufficient、原因和缺失问题。", "不足时，query_rewrite 生成更聚焦的查询。", "重新检索和评分，随后执行矛盾检测。"] },
    output: { en: "EvidenceAssessment, optional QueryRewrite and validated ContradictionRecord entries.", zh: "EvidenceAssessment、可选 QueryRewrite，以及校验后的 ContradictionRecord。" },
    implementation: ["EvidenceAssessment", "QueryRewrite", "ContradictionRecord", "global evidence allowlist"],
    guards: { en: ["Every repair consumes reserved search budget.", "Contradiction evidence IDs must be globally allowed."], zh: ["每次修复都消耗预留检索预算。", "矛盾记录中的证据 ID 必须在全局白名单内。"] },
    proof: { en: "The saved run reached 3/3 sufficient assessments, so no rewrite was needed and no contradiction was found.", zh: "真实运行达到 3/3 证据充分，因此无需重写，也未发现矛盾。" },
  },
  {
    id: "memory-report", number: "07", lane: "runtime", title: { en: "Memory Fold & Final Report", zh: "记忆压缩与最终报告" }, subtitle: "bounded context → cited answer",
    summary: { en: "The system compresses the trace, then writes only against an explicit evidence-ID allowlist.", zh: "系统先压缩轨迹，再仅依据显式 evidence-ID 白名单撰写答案。" },
    input: { en: "Accepted evidence, assessments and contradiction records.", zh: "已接受证据、充分性评估与矛盾记录。" },
    process: { en: ["memory_fold produces a bounded summary.", "report receives the summary and allowed_evidence_ids.", "The runtime validates every returned citation before persistence."], zh: ["memory_fold 生成有长度上限的摘要。", "report 接收摘要与 allowed_evidence_ids。", "Runtime 在持久化前校验每个返回引用。"] },
    output: { en: "final_report and cited_evidence_ids, or a safe insufficient-evidence response.", zh: "final_report 与 cited_evidence_ids，或安全的证据不足答复。" },
    implementation: ["MemoryFold", "report stage contract", "final_evidence_ids"],
    guards: { en: ["Citations must be a subset of the allowlist.", "A substantive report must cite at least one evidence item."], zh: ["引用必须是白名单子集。", "有实质内容的报告必须至少引用一条证据。"] },
    proof: { en: "The real run completed memory fold and persisted a cited final report.", zh: "真实运行完成 Memory Fold，并持久化带引用的最终报告。" },
  },
  {
    id: "persistence-recovery", number: "08", lane: "reliability", title: { en: "Persistence, Cancel & Resume", zh: "持久化、取消与恢复" }, subtitle: "SQLite WAL + checkpoints",
    summary: { en: "Durability is a first-class control path, not an afterthought around the happy path.", zh: "持久化是第一等控制链路，而不是只围绕成功路径的补丁。" },
    input: { en: "Every state mutation, tool result and logical stage completion.", zh: "每次状态变化、工具结果与逻辑阶段完成事件。" },
    process: { en: ["Persist the full aggregate as payload_json in SQLite/WAL.", "Increment RunCheckpoint.version at logical boundaries.", "BEGIN IMMEDIATE records cancel; resume reloads and skips completed stages."], zh: ["把完整聚合对象以 payload_json 写入 SQLite/WAL。", "在逻辑边界递增 RunCheckpoint.version。", "BEGIN IMMEDIATE 写入取消；resume 重载并跳过已完成阶段。"] },
    output: { en: "An idempotently resumable run with complete error, usage and stop-reason history.", zh: "可幂等恢复，并保留完整错误、用量和停止原因历史的 Run。" },
    implementation: ["AgenticRunRepository", "SQLite WAL", "RunCheckpoint", "completed_stages", "cancel_requested"],
    guards: { en: ["Five hard limits: wall time, agent steps, searches, page reads and tokens.", "Cooperative cancellation is checked between durable steps."], zh: ["五类硬限制：时长、Agent 步数、搜索、读页与 Token。", "在持久化步骤之间检查协作式取消。"] },
    proof: { en: "The accepted run persisted checkpoint version 14 and passed database round-trip checks.", zh: "验收运行持久化到 checkpoint v14，并通过数据库往返一致性检查。" },
  },
  {
    id: "offline-training", number: "09", lane: "training", title: { en: "Offline Training Experiment", zh: "离线训练实验" }, subtitle: "Teacher traces → 2×4090 → Qwen LoRA",
    summary: { en: "A completed Qwen3-8B LoRA experiment remains as training evidence, but the current interview demo calls DeepSeek directly and does not serve this adapter.", zh: "已完成的 Qwen3-8B LoRA 实验作为训练证据保留；当前面试演示直接调用 DeepSeek，不再部署这个 Adapter。" },
    input: { en: "500 teacher trajectories following the four-action protocol.", zh: "500 条遵循四动作协议的 Teacher 轨迹。" },
    process: { en: ["Validate and filter: 500 → 479 usable.", "Split 431 train / 48 validation.", "SSH to an 8× RTX 4090 node; train on physical GPU 4 + 5 with FSDP2 BF16 LoRA."], zh: ["校验与过滤：500 → 479 条可用。", "划分 431 条训练 / 48 条验证。", "通过 SSH 连接 8× RTX 4090 节点；在物理 GPU 4 + 5 上用 FSDP2 BF16 LoRA 训练。"] },
    output: { en: "Qwen3-8B LoRA adapter (rank 16, alpha 32) plus held-out evaluation artifacts.", zh: "Qwen3-8B LoRA Adapter（rank 16、alpha 32）及独立评估产物。" },
    implementation: ["search / read_page / evaluate_evidence / answer", "FSDP2", "BF16", "LoRA r=16 α=32", "90 held-out questions"],
    guards: { en: ["1–8 steps; exactly one answer and it must be last.", "Non-answer steps require observations; evidence IDs must come from the seed.", "The page exposes topology, never SSH secrets."], zh: ["轨迹 1–8 步；answer 恰好一次且必须最后。", "非 answer 步必须有 observation；证据 ID 必须来自 Seed。", "页面只展示拓扑，绝不展示 SSH 密钥。"] },
    proof: { en: "The short 10-step run validated the two-GPU pipeline; it is historical evidence, not a dependency of the current web runtime or a final-convergence claim.", zh: "10 个优化步骤验证了双卡训练链路；它是历史实验依据，不是当前网页运行依赖，也不宣称最终收敛。" },
  },
];

export function getArchitectureNode(id?: string) {
  return architectureNodes.find((node) => node.id === id) ?? architectureNodes[1];
}
