import assert from "node:assert/strict";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("deepseek-smoke", `${Date.now()}`);
const { default: worker } = await import(workerUrl.href);
const savedRuns = [];
const database = {
  prepare(sql) {
    let values = [];
    return { bind(...next) { values = next; return this; }, async run() { if (sql.startsWith("INSERT INTO live_runs")) savedRuns.push(values); return { success: true, results: [] }; } };
  },
};

const actions = [
  { rationale_summary:"Find retrieval and fusion evidence.",action:"search",arguments:{query:"BM25 Dense Retrieval RRF"},evidence_ids:[],final_answer:null },
  { rationale_summary:"Read the dense retrieval passage.",action:"read_page",arguments:{evidence_id:"doc-002"},evidence_ids:["doc-002"],final_answer:null },
  { rationale_summary:"Read the rank fusion passage.",action:"read_page",arguments:{evidence_id:"doc-003"},evidence_ids:["doc-003"],final_answer:null },
  { rationale_summary:"Evaluate the two passages.",action:"evaluate_evidence",arguments:{},evidence_ids:["doc-002","doc-003"],final_answer:null },
  { rationale_summary:"Evidence is sufficient.",action:"answer",arguments:{},evidence_ids:["doc-002","doc-003"],final_answer:"Write the grounded report." },
];
let call = 0;
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => {
  if (!String(url).startsWith("https://deepseek-smoke.invalid")) return originalFetch(url, init);
  const content = call === 0 ? JSON.stringify({route:"evidence_research",category:"technical_analysis",confidence:.99,reason:"The comparison needs evidence.",final_answer:null}) : call <= actions.length ? JSON.stringify(actions[call-1]) : JSON.stringify({ final_report:"Dense retrieval provides semantic matching [doc-002], and RRF combines ranks without adding incompatible raw scores [doc-003].",cited_evidence_ids:["doc-002","doc-003"] });
  call += 1;
  return Response.json({ model:"deepseek-v4-flash-smoke",choices:[{message:{content}}],usage:{prompt_tokens:10,completion_tokens:10,total_tokens:20} });
};

try {
  const pending = [];
  const response = await worker.fetch(new Request("http://localhost/api/live-research/runs", { method:"POST",headers:{"content-type":"application/json",cookie:"live_demo_access=deepseek-smoke-access"},body:JSON.stringify({query:"Why do BM25, Dense Retrieval and RRF work together?"}) }), { ASSETS:{fetch:async()=>new Response("Not found",{status:404})},DB:database,DEEPSEEK_API_KEY:"smoke-only",DEEPSEEK_BASE_URL:"https://deepseek-smoke.invalid",DEEPSEEK_MODEL:"deepseek-v4-flash-smoke",LIVE_DEMO_ACCESS_TOKEN:"deepseek-smoke-access" }, { waitUntil(promise){pending.push(promise);},passThroughOnException(){} });
  const stream = await response.text(); await Promise.all(pending);
  const events = stream.split("\n\n").map((chunk)=>chunk.split("\n").find((line)=>line.startsWith("data: "))).filter(Boolean).map((line)=>JSON.parse(line.slice(6)));
  const completed = events.findLast((event)=>event.type==="completed");
  assert.equal(response.status,200); assert.ok(completed); assert.equal(completed.status,"completed");
  assert.ok(events.filter((event)=>event.payload?.contract==="agent_action"&&event.payload?.actor==="deepseek").length>=5);
  assert.ok(completed.payload.evidence.length>=2); assert.equal(savedRuns.length,1); assert.equal(call,7);
  console.log(JSON.stringify({status:"passed",events:events.length,deepseek_calls:call,evidence_count:completed.payload.evidence.length}));
} finally { globalThis.fetch = originalFetch; }
