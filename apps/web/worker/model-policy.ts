export interface AgentAction {
  rationale_summary: string;
  action: "search" | "read_page" | "evaluate_evidence" | "answer";
  arguments: Record<string, unknown>;
  evidence_ids: string[];
  final_answer: string | null;
}

export interface PolicyStep {
  step: number;
  action: AgentAction;
  status: "succeeded" | "failed";
  observation: Record<string, unknown>;
  error_code?: string;
}

export interface PolicyState {
  question: string;
  history: PolicyStep[];
  discovered_evidence_ids: string[];
  read_evidence_ids: string[];
  evaluated_evidence_ids: string[];
  remaining_actions: number;
}

export const ACTION_SYSTEM_PROMPT = (
  "You are the DeepTrace DeepSeek action policy. Return exactly one JSON object " +
  "and no markdown. Schema: {\"rationale_summary\":string,\"action\":" +
  "\"search|read_page|evaluate_evidence|answer\",\"arguments\":object," +
  "\"evidence_ids\":[string],\"final_answer\":string|null}. Choose only " +
  "the next action. Tool observations are supplied by the deterministic runtime; " +
  "never invent an observation or evidence ID. Use answer only after evidence " +
  "has been read and evaluated. Keep final_answer to one short handoff sentence; " +
  "the same DeepSeek model is called again as a grounded writer after runtime validation."
);

export function buildPolicyMessages(state: PolicyState): Array<{ role: string; content: string }> {
  const userContent = [
    `Question: ${state.question}`,
    `Remaining actions including answer: ${state.remaining_actions}`,
    `Evidence IDs returned by search: ${JSON.stringify(state.discovered_evidence_ids)}`,
    `Evidence IDs successfully read and allowed in answer: ${JSON.stringify(state.read_evidence_ids)}`,
    `Evidence IDs explicitly evaluated: ${JSON.stringify(state.evaluated_evidence_ids)}`,
    "Minimum distinct evidence required before answer: 2",
    "evaluate_evidence required before answer: true",
    `Executed history (runtime observations are authoritative): ${JSON.stringify(state.history)}`,
    state.remaining_actions <= 1
      ? "This is the final allowed action. Return answer using only successfully read evidence IDs."
      : "Choose one next action. Prefer search, then read_page, then evaluate_evidence, then answer.",
  ].join("\n");
  return [
    { role: "system", content: ACTION_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ];
}

export function parseAgentAction(content: string): AgentAction {
  const stripped = stripCodeFence(content);
  let parsed: unknown;
  try { parsed = JSON.parse(stripped); } catch { throw new Error("DeepSeek policy returned invalid JSON."); }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Policy action must be a JSON object.");
  const value = parsed as Record<string, unknown>;
  const allowed = new Set(["search", "read_page", "evaluate_evidence", "answer"]);
  if (typeof value.rationale_summary !== "string" || value.rationale_summary.trim().length < 3) throw new Error("Policy action is missing rationale_summary.");
  if (typeof value.action !== "string" || !allowed.has(value.action)) throw new Error("Policy action name is invalid.");
  if (value.arguments !== undefined && (!value.arguments || typeof value.arguments !== "object" || Array.isArray(value.arguments))) throw new Error("Policy action arguments must be an object.");
  if (value.evidence_ids !== undefined && (!Array.isArray(value.evidence_ids) || value.evidence_ids.some((id) => typeof id !== "string"))) throw new Error("Policy evidence_ids must be strings.");
  const action = value.action as AgentAction["action"];
  const finalAnswer = typeof value.final_answer === "string" ? value.final_answer.trim() : null;
  if (action === "answer" && !finalAnswer) throw new Error("Answer action requires final_answer.");
  if (action !== "answer" && finalAnswer) throw new Error("Only the answer action may return final_answer.");
  return {
    rationale_summary: value.rationale_summary.trim().slice(0, 500),
    action,
    arguments: (value.arguments || {}) as Record<string, unknown>,
    evidence_ids: (value.evidence_ids || []) as string[],
    final_answer: finalAnswer,
  };
}

function stripCodeFence(content: string): string {
  const value = content.trim();
  if (!value.startsWith("```") || !value.endsWith("```")) return value;
  return value.split(/\r?\n/).slice(1, -1).join("\n").trim();
}
