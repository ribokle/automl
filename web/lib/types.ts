export const AGENT_ORDER = [
  "ingestion",
  "ppg_mapping",
  "ppg_selection",
  "feature_selection",
  "eda",
  "feature_engineering",
  "feature_refine",
  "modeling",
  "results_reasoning",
  "decomposition",
  "simulation",
  "optimization",
  "validation",
  "insights",
] as const;

export type AgentName = (typeof AGENT_ORDER)[number];

export type AgentStatus =
  | "pending"
  | "running"
  | "awaiting_approval"
  | "done"
  | "failed"
  | "skipped";

export interface RunEvent {
  ts: string;
  run_id: string;
  type: string;
  agent?: AgentName;
  status?: AgentStatus;
  outputs?: Record<string, unknown>;
  tool?: string;
  rows?: number;
  checks?: number;
  columns?: number;
  n_skus?: number;
  n_ppgs?: number;
  error?: string;
  approved?: boolean;
}

export interface AgentArtifact {
  id: string;
  name: string;
  path: string;
  mime: string;
  agent: string;
  created_at: string;
}

export interface AgentState {
  agent: AgentName;
  status: AgentStatus;
  confidence: number;
  reasoning: string;
  outputs?: Record<string, unknown> | null;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  artifacts?: AgentArtifact[];
}

export interface RunStateFull {
  id: string;
  status: string;
  data_path: string;
  run_dir: string;
  created_at: string;
  agents: Record<AgentName, AgentState>;
  gates: Record<string, boolean>;
}

export interface RunSummary {
  id: string;
  status: string;
  data_path: string;
  run_dir: string;
  created_at: string;
}

export interface PPGRow {
  ppg_id: string;
  sku: string;
  brand: string;
  category: string;
  pack_size: string;
  median_price: number;
  confidence: number;
  rationale: string;
  flagged: boolean;
}

export interface PPGSelectionRow {
  ppg_id: string;
  n_skus: number;
  total_units: number;
  coverage: number;
  price_cv: number;
  promo_weeks_pct: number;
  score: number;
  eligible: boolean;
  reasoning: string;
  exec_rationale: string;
}
