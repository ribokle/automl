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
  error?: string;
  approved?: boolean;
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
