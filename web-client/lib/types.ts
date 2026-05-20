export type ColorOption = "vault" | "marquee";

export interface Kpi {
  label: string;
  value: string;
  delta?: string;
  positive?: boolean;
  hint?: string;
}

export interface Recommendation {
  ppg_id: string;
  ppg_name: string;
  category: string;
  current_price: number;
  proposed_price: number;
  delta_pct: number;
  unit_lift_pct: number;
  revenue_lift_usd: number;
  confidence: number;
  elasticity: number;
  rationale: string;
  flagged?: boolean;
}

export interface WeeklyPoint {
  week: string;
  baseline_revenue: number;
  proposed_revenue: number;
  baseline_units: number;
  proposed_units: number;
}

export interface ElasticityPoint {
  price: number;
  units: number;
  revenue: number;
}

export interface ValidationCheck {
  name: string;
  status: "pass" | "warn" | "fail";
  metric: string;
  note: string;
}

export interface PPGForestPoint {
  ppg_id: string;
  elasticity: number;
  ci_low: number;
  ci_high: number;
  benchmark_low?: number;
  benchmark_high?: number;
  benchmark_mean?: number;
  benchmark_source?: string;
  benchmark_status?: "in_band" | "out_band_low" | "out_band_high" | "no_benchmark";
}

export interface MethodologyStep {
  agent: string;
  title: string;
  description: string;
  implication: string;
}

export interface RunSummary {
  id: string;
  status: string;
  created_at: string;
  archived?: boolean;
}

export interface ClientPayload {
  run_id: string | null;
  generated_at: string;
  kpis: Kpi[];
  anchor: {
    label: string;
    value: string;
    delta: string;
    detail: string;
  };
  weekly: WeeklyPoint[];
  recommendations: Recommendation[];
  elasticity_by_ppg: Record<string, ElasticityPoint[]>;
  validation: ValidationCheck[];
  trust_score: number;
  forest: PPGForestPoint[];
  methodology: MethodologyStep[];
  narrative: string;
}
