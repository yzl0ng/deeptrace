"use client";

import { useEffect, useRef, useState } from "react";

const liveApiRoot = "/api/live-research";
type LiveEvent = { seq:number; type:string; status:string; title:string; detail:string; elapsed_ms:number; payload?:Record<string,unknown> };
type LiveEvidence = { id:string; title:string; topic:string; excerpt:string; score:number; matched_terms:string[] };
type LiveUsage = { total_tokens?:number };
type LiveUsageByModel = { deepseek?:LiveUsage };
function prettyJson(value:unknown){try{return JSON.stringify(value,null,2);}catch{return String(value);}}
const liveStatusLabels = {
  en: { queued:"Queued", scoping:"Scoping", routing:"Complexity route", planning:"Planning", researching:"Researching", checking_evidence:"Evidence gate", compressing:"Memory fold", writing:"Writing", completed:"Completed", failed:"Failed" },
  zh: { queued:"已排队", scoping:"范围澄清", routing:"复杂度路由", planning:"研究规划", researching:"并行检索", checking_evidence:"证据门控", compressing:"记忆压缩", writing:"撰写报告", completed:"已完成", failed:"失败" },
};
const liveStepExplanations = {
  en: { queued:"The API validates the request and creates the durable ResearchRun identity.", scoping:"The runtime normalizes the question, fixes its trust boundary and decides whether clarification is required.", routing:"DeepSeek proposes direct_answer or evidence_research, then the Runtime applies a conservative gate. Only arithmetic and user-supplied text transformations may bypass search; uncertainty defaults to research.", planning:"The planning policy must return a schema-valid list of bounded research units.", researching:"The Supervisor dispatches a real local_search action and records the passages returned by the tool.", checking_evidence:"A deterministic gate checks whether enough retrieved evidence can enter the citation allowlist.", compressing:"Accepted observations are folded into bounded memory before the next model call.", writing:"For a direct route this is the answer returned by the routing API call; for research it is constrained by the evidence allowlist.", completed:"The runtime packages the trace, evidence, report and token usage for durable persistence.", failed:"The runtime stopped and returned a sanitized error event." },
  zh: { queued:"API 校验请求并创建可持久化的 ResearchRun 身份。", scoping:"Runtime 规范化问题、固定信任边界，并判断是否需要澄清。", routing:"DeepSeek 先建议 direct_answer 或 evidence_research，再由 Runtime 保守复核。只有算术和基于用户已提供文本的转换任务可以跳过搜索；拿不准默认研究。", planning:"规划策略必须返回满足 Schema 的有限研究子任务列表。", researching:"Supervisor 发起真实 local_search 动作，并记录工具实际返回的 Passage。", checking_evidence:"确定性门控检查证据是否足够，并决定哪些 Passage 可以进入引用白名单。", compressing:"下一次模型调用前，仅把已通过的观察压缩进有界记忆。", writing:"直答路由展示第一次 API 调用返回的答案；研究路由则只允许根据证据白名单写作。", completed:"Runtime 汇总轨迹、证据、报告和 Token 用量，写入持久化运行记录。", failed:"Runtime 已停止，并返回经过脱敏的错误事件。" },
};
const liveActorLabels = {
  en: { deepseek:"DEEPSEEK · POLICY / WRITER", runtime:"DETERMINISTIC RUNTIME", tool:"SEARCH TOOL" },
  zh: { deepseek:"DEEPSEEK · 决策 / 写作", runtime:"确定性 RUNTIME", tool:"检索工具" },
};
const stages = {
  en: [
    ["01","Clarify","scope locked","done"], ["02","Plan","3 subtasks","done"], ["03","Search","3 tool calls","parallel"],
    ["04","Assess","3 / 3 sufficient","pass"], ["05","Synthesize","evidence linked","done"], ["06","Report","protocol valid","done"],
  ],
  zh: [
    ["01","澄清","范围已锁定","完成"], ["02","规划","3 个子任务","完成"], ["03","检索","3 次工具调用","并行"],
    ["04","评估","3 / 3 充分","通过"], ["05","综合","证据已关联","完成"], ["06","报告","协议有效","完成"],
  ],
};
const stageCopy = {
  en: [
    ["No clarification needed","The query is specific enough to establish an executable research brief."],
    ["Research plan committed","Split the question into complementarity, raw-score mismatch and RRF fusion."],
    ["Three research units executed","The Supervisor dispatches three parallel local_search calls for the three plan items."],
    ["Evidence sufficiency passed","All three subtasks pass the sufficiency gate on attempt one; their tool-call IDs enter the evidence allowlist."],
    ["Claims mapped to evidence","The runtime folds observations into memory and binds each final claim to an allowlisted tool call."],
    ["Final protocol satisfied","The report, three evidence IDs, usage and checkpoint v14 are persisted with no runtime error."],
  ],
  zh: [
    ["无需追加澄清","问题信息足以形成可执行的研究简报。"],
    ["研究计划已提交","将问题拆为检索互补性、原始分数量纲差异和 RRF 融合三个子任务。"],
    ["三个研究单元并行执行","Supervisor 针对三个计划项并行发起三次 local_search 调用。"],
    ["证据充分性检查通过","三个子任务均在第一次尝试中通过充分性门禁，对应工具调用编号进入证据白名单。"],
    ["主张已映射到证据","运行时将观察结果折叠进记忆，并把最终主张绑定到白名单工具调用。"],
    ["最终协议已满足","报告、三条证据编号、用量与检查点 v14 均已持久化，运行时错误为 0。"],
  ],
};
const evidence = {
  en: [
    ["tool:399b","BM25 + Dense notes","ALLOWLISTED","Lexical and semantic retrieval complementarity."],
    ["tool:68da","Score-scale evidence","ALLOWLISTED","Raw BM25 and cosine scores are not directly comparable."],
    ["tool:ca25","RRF fusion record","ALLOWLISTED","Reciprocal-rank fusion removes raw-scale bias."],
  ],
  zh: [
    ["tool:399b","BM25 + Dense 记录","白名单","支持词法检索与语义检索的互补关系。"],
    ["tool:68da","分数量纲证据","白名单","说明 BM25 原始分数与余弦相似度不能直接比较。"],
    ["tool:ca25","RRF 融合记录","白名单","支持用倒数排名融合消除原始量纲偏差。"],
  ],
};

const replayCalls = {
  en: [
    { fn:"createResearchRun()", role:"API entry", input:{query:"Why are BM25 and Dense Retrieval complementary?",max_actions:24}, output:{run_id:"run-0738372d…",status:"scoping"}, effect:"Validates the request and allocates a durable run identity." },
    { fn:"invokePolicy('plan')", role:"DeepSeek structured call", input:{stage:"plan",allowed_actions:["search","read_page","evaluate_evidence","answer"]}, output:{subtasks:3,contract_valid:true}, effect:"Turns the question into bounded research units; JSON Schema keeps the model output executable." },
    { fn:"executeTool('local_search')", role:"Runtime tool dispatch", input:{queries:3,top_k:5,parallel:true}, output:{tool_calls:3,passages_returned:9}, effect:"Executes BM25 outside the model and records the actual observations with stable IDs." },
    { fn:"gradeEvidence()", role:"Deterministic gate", input:{candidate_sets:3,minimum_passages:1}, output:{sufficient:"3 / 3",rewrite_required:false}, effect:"Prevents weak or invented evidence from entering the citation allowlist." },
    { fn:"foldMemory()", role:"Context control", input:{observations:9,allowlisted_calls:3}, output:{bounded_memory:true,contradictions:0}, effect:"Compresses accepted observations so the final model call receives a small, traceable context." },
    { fn:"writeGroundedReport()", role:"DeepSeek grounded call", input:{allowed_evidence_ids:["tool:399b","tool:68da","tool:ca25"]}, output:{status:"completed",citations:3,checkpoint:"v14"}, effect:"Writes only from allowlisted evidence, validates every citation, then persists the result." },
  ],
  zh: [
    { fn:"createResearchRun()", role:"API 入口", input:{query:"为什么 BM25 与 Dense Retrieval 互补？",max_actions:24}, output:{run_id:"run-0738372d…",status:"scoping"}, effect:"校验请求并创建可持久化的运行身份，后续每一步都归属这个 run_id。" },
    { fn:"invokePolicy('plan')", role:"DeepSeek 结构化调用", input:{stage:"plan",allowed_actions:["search","read_page","evaluate_evidence","answer"]}, output:{subtasks:3,contract_valid:true}, effect:"把问题拆成有界研究单元；JSON Schema 让模型输出可以被程序安全执行。" },
    { fn:"executeTool('local_search')", role:"Runtime 工具调度", input:{queries:3,top_k:5,parallel:true}, output:{tool_calls:3,passages_returned:9}, effect:"模型只提议搜索，真正的 BM25 调用由 Runtime 执行，并记录稳定证据编号。" },
    { fn:"gradeEvidence()", role:"确定性证据门控", input:{candidate_sets:3,minimum_passages:1}, output:{sufficient:"3 / 3",rewrite_required:false}, effect:"证据不足就阻止写作并触发查询重写，避免模型凭记忆补答案。" },
    { fn:"foldMemory()", role:"上下文管理", input:{observations:9,allowlisted_calls:3}, output:{bounded_memory:true,contradictions:0}, effect:"只压缩已通过门控的观察，让最终模型调用拿到短小、可追溯的上下文。" },
    { fn:"writeGroundedReport()", role:"DeepSeek 证据写作", input:{allowed_evidence_ids:["tool:399b","tool:68da","tool:ca25"]}, output:{status:"completed",citations:3,checkpoint:"v14"}, effect:"只使用白名单证据写报告，逐条验证引用后再持久化最终结果。" },
  ],
};

const functionGuide = {
  en: [
    ["routeQuery()","DeepSeek proposes a route, then Runtime rejects unsafe direct answers. Only closed-form or user-supplied transformations bypass search.","Conservative route"],
    ["runPolicyLoop()","Calls DeepSeek once per decision and keeps the action history within an eight-action budget.","Typed action"],
    ["callModel()","Sends a server-side DeepSeek request; the API key never enters the browser.","JSON response + tokens"],
    ["executePolicyAction()","Validates the proposed action and dispatches only an allowed runtime tool.","Real observation"],
    ["bm25Search()","Ranks the pinned demo corpus and returns matched terms plus stable Evidence IDs.","Top-k passages"],
    ["gradeEvidence()","Checks read evidence against minimum sufficiency and the citation allowlist.","Pass / repair"],
    ["parseReport() + persist","Validates report JSON and citations, then saves the complete SSE trace.","Auditable run"],
  ],
  zh: [
    ["routeQuery()","DeepSeek 先提出路由，Runtime 再拒绝不安全的直答；只有封闭计算或用户已提供文本的转换任务跳过搜索。","保守路由"],
    ["runPolicyLoop()","每次只让 DeepSeek 决定一个动作，并把动作历史限制在 8 步预算内。","强类型动作"],
    ["callModel()","从服务端调用 DeepSeek，API Key 永远不会进入浏览器。","JSON 返回 + Token"],
    ["executePolicyAction()","校验模型提议，只允许 Runtime 执行白名单内的工具动作。","真实 observation"],
    ["bm25Search()","在固定演示语料上排序，返回命中词和稳定 Evidence ID。","Top-k Passage"],
    ["gradeEvidence()","检查已读证据是否充分，以及引用是否属于白名单。","通过 / 修复"],
    ["parseReport() + persist","校验报告 JSON 和引用，再保存完整 SSE 轨迹。","可审计运行"],
  ],
};

export function Workbench({ lang = "en" }: { lang?: "en" | "zh" }) {
  const zh = lang === "zh"; const s = stages[lang]; const copy = stageCopy[lang]; const ev = evidence[lang];
  const [stage,setStage] = useState(4); const [selected,setSelected] = useState(0); const [mode,setMode] = useState<"replay"|"live">("replay");
  const replayCall = replayCalls[lang][stage];
  const [query,setQuery] = useState(zh ? "为什么 BM25 与 Dense Retrieval 互补？为什么 RRF 不应直接相加两路原始分数？" : "Why are BM25 and Dense Retrieval complementary, and why should RRF not directly sum their raw scores?");
  const [liveState,setLiveState] = useState<"checking"|"locked"|"ready"|"offline"|"running"|"done"|"error">("checking");
  const [liveEvents,setLiveEvents] = useState<LiveEvent[]>([]); const [liveMessage,setLiveMessage] = useState("");
  const [liveEvidence,setLiveEvidence] = useState<LiveEvidence[]>([]); const [liveReport,setLiveReport] = useState("");
  const [liveRunId,setLiveRunId] = useState(""); const [liveUsage,setLiveUsage] = useState<LiveUsage|null>(null); const [liveUsageByModel,setLiveUsageByModel] = useState<LiveUsageByModel|null>(null);
  const [liveAccessCode,setLiveAccessCode] = useState("");
  const [selectedLiveSeq,setSelectedLiveSeq] = useState<number|null>(null);
  const followLiveRef=useRef(true);

  useEffect(() => { fetch(`${liveApiRoot}/status`).then(async response => {
    if (!response.ok) throw new Error("Status endpoint unavailable"); const status = await response.json() as {ready?:boolean;authorized?:boolean}; setLiveState(status.ready ? (status.authorized ? "ready" : "locked") : "offline");
    if (!status.ready) setLiveMessage(zh ? "实时服务已响应，但模型密钥或持久化数据库尚未就绪。" : "The live service responded, but its model key or durable database is not ready.");
  }).catch(() => { setLiveState("offline"); setLiveMessage(zh ? "当前无法连接同源实时运行服务。" : "The same-origin live runtime is currently unreachable."); }); },[zh]);

  async function unlockLiveMode(){setLiveMessage("");try{const response=await fetch(`${liveApiRoot}/unlock`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({token:liveAccessCode.trim()})});const payload=await response.json() as {message?:string};if(!response.ok)throw new Error(payload.message??(zh?"演示访问码不正确。":"The demo access code is incorrect."));setLiveAccessCode("");setLiveState("ready");}catch(error){setLiveMessage(error instanceof Error?error.message:(zh?"解锁失败。":"Unlock failed."));}}

  async function startLiveRun(){ if(!query.trim())return; followLiveRef.current=true;setLiveState("running");setLiveMessage("");setLiveEvents([]);setSelectedLiveSeq(null);setLiveEvidence([]);setLiveReport("");setLiveRunId("");setLiveUsage(null);setLiveUsageByModel(null);try{
    const response=await fetch(`${liveApiRoot}/runs`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({query:query.trim()})});
    if(!response.ok){const payload=await response.json() as {message?:string};if(response.status===401)setLiveState("locked");throw new Error(payload.message??(zh?"研究任务执行失败。":"The research run failed."));}
    if(!response.body)throw new Error(zh?"浏览器没有收到实时事件流。":"The browser did not receive a live event stream.");
    const reader=response.body.getReader();const decoder=new TextDecoder();let buffer="";
    while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const chunks=buffer.split("\n\n");buffer=chunks.pop()??"";
      for(const chunk of chunks){const line=chunk.split("\n").find(item=>item.startsWith("data: "));if(!line)continue;const event=JSON.parse(line.slice(6)) as LiveEvent;setLiveEvents(previous=>[...previous,event]);if(followLiveRef.current)setSelectedLiveSeq(event.seq);
        if(event.type==="accepted"&&event.payload?.run_id)setLiveRunId(String(event.payload.run_id));
        if(event.type==="report"&&event.payload?.final_report)setLiveReport(String(event.payload.final_report));
        if(event.type==="completed"){setLiveState("done");const evidencePayload=event.payload?.evidence;if(Array.isArray(evidencePayload))setLiveEvidence(evidencePayload as LiveEvidence[]);if(event.payload?.final_report)setLiveReport(String(event.payload.final_report));if(event.payload?.usage)setLiveUsage(event.payload.usage as LiveUsage);if(event.payload?.usage_by_model)setLiveUsageByModel(event.payload.usage_by_model as LiveUsageByModel);}
        if(event.type==="error"){setLiveState("error");setLiveMessage(event.detail);}
      }
    }
    setLiveState(previous=>previous==="running"?"done":previous);
  }catch(error){setLiveState("error");setLiveMessage(error instanceof Error?error.message:(zh?"研究任务执行失败。":"The research run failed."));}}

  const selectedLiveEvent=liveEvents.find(event=>event.seq===selectedLiveSeq)??liveEvents.at(-1);
  const selectedPayload=selectedLiveEvent?.payload;
  const selectedInput=selectedPayload?.input??{event_status:selectedLiveEvent?.status??"idle"};
  const selectedOutput=selectedPayload?.output??selectedPayload??{message:selectedLiveEvent?.detail??(zh?"运行后将在这里显示本步返回结果。":"The returned value for each step will appear here after a run.")};
  const selectedContract=String(selectedPayload?.contract??selectedPayload?.action??selectedLiveEvent?.type??"runtime");
  const selectedActor=String(selectedPayload?.actor??"runtime") as keyof typeof liveActorLabels.en;

  return <section id="workbench" className="workbench shell">
    <div className="workbench-top"><div><span className="section-index">{zh?"已验证真实轨迹":"VERIFIED REAL RUN"}</span><h2>{zh?"运行轨迹":"Run trace"} / 0738372d</h2></div><div className="run-meta"><span><b>{zh?"状态":"STATUS"}</b><i className="green">{zh?"已完成":"COMPLETED"}</i></span><span><b>{zh?"运行模型":"RUNTIME MODEL"}</b><i>DEEPSEEK-V4-FLASH</i></span><span><b>{zh?"总用量":"USAGE"}</b><i>9,919 TOKENS</i></span></div></div>
    <div className="mode-switch"><button className={mode==="replay"?"active":""} onClick={()=>setMode("replay")}>{zh?"真实轨迹回放":"Verified real run"}</button><button className={mode==="live"?"active":""} onClick={()=>setMode("live")}>{zh?"实时 API":"Live API"}<i className={liveState==="ready"||liveState==="done"?"online":""}/></button></div>
    <div className="query-bar"><span>{zh?"真实研究问题":"REAL RESEARCH QUERY"}</span>{mode==="replay"?<p>{query}</p>:<input aria-label={zh?"实时研究问题":"Live research query"} value={query} onChange={event=>setQuery(event.target.value)}/>} {mode==="replay"?<button onClick={()=>setStage(0)}>↻ {zh?"回放":"replay"}</button>:<button disabled={!(["ready","done","error"].includes(liveState))} onClick={startLiveRun}>{liveState==="running"?(zh?"执行中…":"running…"):(zh?"开始运行 →":"start run →")}</button>}</div>
    {mode==="replay"?<div className="trace-grid">
      <aside className="stage-list" aria-label="Execution stages"><div className="panel-label">{zh?"执行阶段":"EXECUTION"} / 06</div>{s.map((item,index)=><button key={item[0]} className={stage===index?"selected":""} onClick={()=>setStage(index)}><span className="stage-num">{item[0]}</span><span><b>{item[1]}</b><small>{item[2]}</small></span><time>{item[3]}</time></button>)}<div className="budget"><span>{zh?"步骤预算":"STEP BUDGET"}</span><strong>9 / 24</strong><div><i style={{width:"37.5%"}}/></div><small>{zh?"3 次检索 · 检查点 v14":"3 searches · checkpoint v14"}</small></div></aside>
      <article className="trace-main"><div className="panel-label"><span>{zh?"当前状态":"ACTIVE STATE"}</span><span>{zh?"步骤":"STEP"} {s[stage][0]} / 06</span></div><div className="state-heading"><span>{s[stage][0]}</span><div><small>{s[stage][1]}</small><h3>{copy[stage][0]}</h3></div></div><p className="state-copy">{copy[stage][1]}</p>
      <div className="replay-api-call"><div className="replay-function"><span>{zh?"本步函数":"STEP FUNCTION"}</span><code>{replayCall.fn}</code><small>{replayCall.role}</small><p>{replayCall.effect}</p></div><div className="live-io-grid"><section><span>{zh?"保存请求 INPUT":"SAVED REQUEST INPUT"}</span><pre>{prettyJson(replayCall.input)}</pre></section><section className="returned"><span>{zh?"真实保存返回 RETURNED OUTPUT":"SAVED REAL RETURNED OUTPUT"}</span><pre>{prettyJson(replayCall.output)}</pre></section></div></div>
      <div className="logic-block"><div><span>{zh?"观察":"OBSERVATION"}</span><p>{stage<2?(zh?"问题约束足以建立可执行研究简报。":"The query constraints are sufficient for execution."):(zh?"3 / 3 子任务证据充分，未发现矛盾。":"Evidence is sufficient for 3 / 3 subtasks with no contradiction.")}</p></div><div><span>{zh?"决策":"DECISION"}</span><p>{stage<4?(zh?"在 24 步、8 次检索和 30k token 预算内继续。":"Continue within the 24-step, 8-search and 30k-token budget."):(zh?"仅使用三条白名单工具证据生成最终报告。":"Generate the final report only from three allowlisted tool-call evidence IDs.")}</p></div></div>
      <div className="report-block"><div className="panel-label"><span>{zh?"真实最终报告 / 摘要":"REAL FINAL REPORT / EXCERPT"}</span><span>{zh?"3 条引用":"3 CITATIONS"}</span></div><p>{zh?"BM25 擅长精确词法匹配，Dense Retrieval 擅长语义召回，因此两者互补。两路原始分数不在同一量纲，直接相加会产生偏差；RRF 通过倒数排名完成无量纲融合。":"BM25 handles exact lexical matches while Dense Retrieval provides semantic recall. Their raw scores live on different scales, so direct addition is biased; RRF instead fuses reciprocal ranks."} {ev.map((item,index)=><button key={item[0]} onClick={()=>setSelected(index)}>[{item[0]}]</button>)}</p></div></article>
      <aside className="evidence-panel"><div className="panel-label"><span>{zh?"真实证据白名单":"REAL EVIDENCE ALLOWLIST"}</span><span>03 / 03</span></div>{ev.map((item,index)=><button key={item[0]} className={selected===index?"selected":""} onClick={()=>setSelected(index)}><div><b>{item[0]}</b><span>{item[2]}</span></div><h4>{item[1]}</h4><p>{item[3]}</p><small>{zh?"充分性":"SUFFICIENCY"} <strong>PASS</strong></small></button>)}<div className="ledger-note">{zh?"数据来自已保存的 run-0738372d…；调用编号在公开页面截短显示。":"Data comes from saved run-0738372d…; call IDs are shortened for public display."}</div></aside>
    </div>:<div className="live-trace-grid">
      <aside className="live-event-list"><div className="panel-label"><span>{zh?"实时状态事件 · 点击查看":"LIVE STATE EVENTS · CLICK TO INSPECT"}</span><span>{liveEvents.length.toString().padStart(2,"0")}</span></div>{liveEvents.length?liveEvents.map(event=>{const actor=String(event.payload?.actor??"runtime") as keyof typeof liveActorLabels.en;return <button key={event.seq} className={`${selectedLiveEvent?.seq===event.seq?"selected":""} ${event.type==="completed"?"complete":""}`} aria-pressed={selectedLiveEvent?.seq===event.seq} onClick={()=>{followLiveRef.current=event.seq===liveEvents.at(-1)?.seq;setSelectedLiveSeq(event.seq);}}><span>{event.seq.toString().padStart(2,"0")}</span><div><b>{liveStatusLabels[lang][event.status as keyof typeof liveStatusLabels.en]??event.status}</b><small><em>{liveActorLabels[lang][actor]??actor}</em>{event.title}</small></div><time>{(event.elapsed_ms/1000).toFixed(1)}s</time></button>}):<div className="live-empty">{liveState==="ready"?(zh?"输入问题并开始运行；每个状态出现后都可以点开查看输入与返回结果。":"Enter a question and start a run; every state can be opened to inspect its input and returned output."):(zh?"正在检查实时运行时…":"Checking the live runtime…")}</div>}</aside>
      <article className="live-result-panel"><div className="panel-label"><span>{selectedLiveEvent?(zh?"步骤返回结果":"STEP RETURN VALUE"):(zh?"当前执行":"ACTIVE EXECUTION")}</span><span>{liveRunId||"—"}</span></div>{selectedLiveEvent?<div className="live-step-inspector"><div className="live-step-head"><span>{zh?"步骤":"STEP"} {selectedLiveEvent.seq.toString().padStart(2,"0")}</span><time>+{selectedLiveEvent.elapsed_ms} ms</time></div><span className={`live-actor ${selectedActor}`}>{liveActorLabels[lang][selectedActor]??selectedActor}</span><span className="live-contract">{selectedPayload?.contract?(zh?"模型契约":"MODEL CONTRACT"):(zh?"运行时动作":"RUNTIME ACTION")} · {selectedContract}</span><h3>{selectedLiveEvent.title}</h3><p className="live-step-detail">{selectedLiveEvent.detail}</p><div className="live-step-explanation"><b>{zh?"这一步在 Agent 中做什么":"WHAT THIS STEP DOES IN THE AGENT"}</b><p>{liveStepExplanations[lang][selectedLiveEvent.status as keyof typeof liveStepExplanations.en]??selectedLiveEvent.detail}</p></div><div className="live-io-grid"><section><span>{zh?"本步输入 INPUT":"STEP INPUT"}</span><pre>{prettyJson(selectedInput)}</pre></section><section className="returned"><span>{zh?"真实返回结果 RETURNED OUTPUT":"REAL RETURNED OUTPUT"}</span><pre>{prettyJson(selectedOutput)}</pre></section></div><details className="live-raw"><summary>{zh?"展开查看原始 SSE 事件 JSON":"OPEN RAW SSE EVENT JSON"}</summary><pre>{prettyJson(selectedLiveEvent)}</pre></details></div>:<div className="live-result-copy"><span className={`live-signal ${liveState}`}>● {liveState.toUpperCase()}</span><h3>{liveState==="ready"?(zh?"实时后端已经就绪。":"The live backend is ready."):liveState==="locked"?(zh?"这台设备需要先解锁。":"Unlock live mode on this device."):(zh?"实时链路当前不可用。":"The live chain is unavailable.")}</h3><p>{liveMessage||(zh?"链路会依次显示 DeepSeek 动作决策、Runtime 校验、工具真实返回和 DeepSeek 证据写作。":"The trace separates DeepSeek action decisions, Runtime validation, real tool returns and grounded DeepSeek writing.")}</p>{liveState==="locked"&&<div className="live-unlock"><input aria-label={zh?"本机演示访问码":"Device demo access code"} type="password" value={liveAccessCode} onChange={event=>setLiveAccessCode(event.target.value)} placeholder={zh?"输入本机演示访问码":"Enter device demo access code"}/><button disabled={!liveAccessCode.trim()} onClick={unlockLiveMode}>{zh?"解锁本机实时模式 →":"unlock live mode →"}</button><small>{zh?"解锁 Cookie 仅保存在这台浏览器 30 天；没有运行次数限制。":"The unlock cookie stays in this browser for 30 days; runs are unlimited."}</small></div>}</div>}
      {liveReport&&<div className="live-final-report"><span>{zh?"证据约束的最终报告":"EVIDENCE-GROUNDED FINAL REPORT"}</span><p>{liveReport}</p><small>{zh?`总用量 ${liveUsage?.total_tokens??0} · DeepSeek ${liveUsageByModel?.deepseek?.total_tokens??0} tokens · ${liveEvidence.length} 条证据`:`${liveUsage?.total_tokens??0} total · DeepSeek ${liveUsageByModel?.deepseek?.total_tokens??0} tokens · ${liveEvidence.length} evidence records`}</small></div>}</article>
      <aside className="live-evidence-panel"><div className="panel-label"><span>{zh?"实时证据白名单":"LIVE EVIDENCE ALLOWLIST"}</span><span>{liveEvidence.length.toString().padStart(2,"0")}</span></div>{liveEvidence.length?liveEvidence.map(item=><article key={item.id}><div><b>{item.id}</b><span>{item.topic}</span></div><h4>{item.title}</h4><p>{item.excerpt}</p><small>BM25 {item.score.toFixed(3)} · {item.matched_terms.join(" / ")}</small></article>):<div className="live-empty">{zh?"通过证据门控后，允许引用的 Passage 会显示在这里。":"Allowlisted passages appear here after the evidence gate."}</div>}</aside>
    </div>}
    <p className="replay-note"><span>●</span> {mode==="replay"?(zh?"真实保存运行：Supervisor · local_search · DeepSeek v4 Flash · checkpoint v14 · 14/14 验收通过":"SAVED REAL RUN · SUPERVISOR · LOCAL_SEARCH · DEEPSEEK V4 FLASH · CHECKPOINT V14 · 14/14 ACCEPTANCE"):(zh?"当前实时链路只依赖 DeepSeek，不再要求远程 Student 服务。":"THE LIVE CHAIN USES DEEPSEEK AND DOES NOT REQUIRE A REMOTE STUDENT SERVICE.")}</p>
    <div className="function-guide"><div className="function-guide-head"><span className="section-index">{zh?"函数级执行说明":"FUNCTION-LEVEL WALKTHROUGH"}</span><h3>{zh?"每个函数为什么存在，返回什么。":"Why each function exists and what it returns."}</h3></div><div className="function-guide-grid">{functionGuide[lang].map(([name,purpose,result])=><article key={name}><code>{name}</code><p>{purpose}</p><span>{zh?"效果":"RESULT"} · {result}</span></article>)}</div></div>
  </section>;
}
