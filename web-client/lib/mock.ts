import type {
  ClientPayload,
  ElasticityPoint,
  Kpi,
  MethodologyStep,
  PPGForestPoint,
  Recommendation,
  ValidationCheck,
  WeeklyPoint,
} from "./types";

const PPGS: Array<{
  id: string;
  name: string;
  category: string;
  current: number;
  proposed: number;
  elasticity: number;
  confidence: number;
  rationale: string;
}> = [
  {
    id: "ppg_01",
    name: "Cola · 12oz multipack",
    category: "Soda",
    current: 6.49,
    proposed: 6.99,
    elasticity: -1.6,
    confidence: 0.92,
    rationale: "Inelastic at this band; mild lift recovers margin without breaking comp gap.",
  },
  {
    id: "ppg_02",
    name: "Cola · 2L bottle",
    category: "Soda",
    current: 2.49,
    proposed: 2.39,
    elasticity: -2.4,
    confidence: 0.88,
    rationale: "Highly elastic; small price cut wins meaningful unit lift.",
  },
  {
    id: "ppg_03",
    name: "Citrus · 12oz multipack",
    category: "Soda",
    current: 5.99,
    proposed: 6.29,
    elasticity: -1.8,
    confidence: 0.85,
    rationale: "Sister brand confirms direction; cross-elasticity favourable.",
  },
  {
    id: "ppg_04",
    name: "Sparkling water · 8pk",
    category: "Water",
    current: 4.49,
    proposed: 4.79,
    elasticity: -1.2,
    confidence: 0.9,
    rationale: "Steady demand category; competitor priced 80c higher leaves headroom.",
  },
  {
    id: "ppg_05",
    name: "Energy · 16oz single",
    category: "Energy",
    current: 2.99,
    proposed: 3.19,
    elasticity: -0.9,
    confidence: 0.94,
    rationale: "Lowest elasticity in catalog; loyal buyers absorb the change.",
  },
  {
    id: "ppg_06",
    name: "Energy · 4pk",
    category: "Energy",
    current: 9.99,
    proposed: 9.49,
    elasticity: -2.1,
    confidence: 0.82,
    rationale: "Multipack reset opens trade-up from singles.",
  },
  {
    id: "ppg_07",
    name: "Juice · 64oz",
    category: "Juice",
    current: 3.79,
    proposed: 3.99,
    elasticity: -1.5,
    confidence: 0.86,
    rationale: "Recovers margin lost to cost-of-goods drift over Q1.",
  },
  {
    id: "ppg_08",
    name: "Juice · 10pk pouches",
    category: "Juice",
    current: 5.49,
    proposed: 5.29,
    elasticity: -2.0,
    confidence: 0.83,
    rationale: "Promo cadence dampened; light EDLP cut sustains share.",
  },
];

function buildRecommendations(): Recommendation[] {
  return PPGS.map((p) => {
    const delta_pct = (p.proposed - p.current) / p.current;
    const unit_lift_pct = -p.elasticity * delta_pct * 0.92;
    const weekly_units_base = 4200 + (p.id.charCodeAt(p.id.length - 1) % 7) * 380;
    const revenue_lift_usd =
      weekly_units_base * 13 * (p.proposed * (1 + unit_lift_pct) - p.current);
    return {
      ppg_id: p.id,
      ppg_name: p.name,
      category: p.category,
      current_price: p.current,
      proposed_price: p.proposed,
      delta_pct,
      unit_lift_pct,
      revenue_lift_usd,
      confidence: p.confidence,
      elasticity: p.elasticity,
      rationale: p.rationale,
      flagged: p.confidence < 0.84,
    };
  });
}

function buildWeekly(recs: Recommendation[]): WeeklyPoint[] {
  const weeks = 13;
  const baseUnitsTotal = recs.reduce(
    (acc, r) => acc + 4200 + ((r.ppg_id.charCodeAt(r.ppg_id.length - 1) % 7) * 380),
    0,
  );
  const baseRevPerUnit =
    recs.reduce((acc, r) => acc + r.current_price, 0) / recs.length;
  const propRevPerUnit =
    recs.reduce((acc, r) => acc + r.proposed_price * (1 + r.unit_lift_pct), 0) /
    recs.length;
  const propUnitsTotal =
    baseUnitsTotal *
    (recs.reduce((acc, r) => acc + (1 + r.unit_lift_pct), 0) / recs.length);

  const out: WeeklyPoint[] = [];
  for (let w = 0; w < weeks; w++) {
    const seasonal = 1 + Math.sin((w / weeks) * Math.PI) * 0.08;
    const noise = 1 + (Math.sin(w * 1.7) * 0.02);
    out.push({
      week: `W${(w + 1).toString().padStart(2, "0")}`,
      baseline_units: Math.round(baseUnitsTotal * seasonal * noise),
      proposed_units: Math.round(propUnitsTotal * seasonal * noise),
      baseline_revenue: Math.round(baseUnitsTotal * seasonal * noise * baseRevPerUnit),
      proposed_revenue: Math.round(propUnitsTotal * seasonal * noise * propRevPerUnit),
    });
  }
  return out;
}

function buildElasticityCurves(recs: Recommendation[]): Record<string, ElasticityPoint[]> {
  const out: Record<string, ElasticityPoint[]> = {};
  for (const r of recs) {
    const points: ElasticityPoint[] = [];
    for (let i = -20; i <= 20; i += 2) {
      const price = r.current_price * (1 + i / 100);
      const units = 5000 * Math.pow(price / r.current_price, r.elasticity);
      points.push({
        price: Number(price.toFixed(2)),
        units: Math.round(units),
        revenue: Math.round(price * units),
      });
    }
    out[r.ppg_id] = points;
  }
  return out;
}

function buildValidation(): ValidationCheck[] {
  return [
    {
      name: "Sign check",
      status: "pass",
      metric: "8 / 8 PPGs",
      note: "All elasticities recovered with the expected negative sign.",
    },
    {
      name: "Rolling-origin CV",
      status: "pass",
      metric: "WAPE 11.2%",
      note: "Hold-out error inside the 15% target on every fold.",
    },
    {
      name: "Ladder constraint",
      status: "pass",
      metric: "0 violations",
      note: "Price ladder monotonicity preserved across sister packs.",
    },
    {
      name: "Margin floor",
      status: "pass",
      metric: "min 23.4%",
      note: "Every proposed price clears the 18% margin floor.",
    },
    {
      name: "Competitor gap",
      status: "warn",
      metric: "1 PPG at gap",
      note: "Cola · 12oz hits the +0c competitor cap — re-check next week.",
    },
    {
      name: "Decomposition residual",
      status: "pass",
      metric: "0.4% mean",
      note: "Observed units reconcile to driver decomposition within tolerance.",
    },
  ];
}

function buildForest(recs: Recommendation[]): PPGForestPoint[] {
  return recs.map((r) => ({
    ppg_id: r.ppg_id,
    elasticity: r.elasticity,
    ci_low: r.elasticity - (1 - r.confidence) * 1.4,
    ci_high: r.elasticity + (1 - r.confidence) * 1.4,
  }));
}

function buildMethodology(): MethodologyStep[] {
  return [
    {
      agent: "Ingestion",
      title: "We read your weekly panel",
      description:
        "SKU × store × week data is validated against schema, distribution, and relationship checks before anything downstream runs.",
      implication: "Bad data never makes it past step one.",
    },
    {
      agent: "Grouping",
      title: "We grouped SKUs into Price-Pack Groups",
      description:
        "An agent clusters SKUs by brand, pack, and size so we can price related items consistently.",
      implication: "Sister packs move together — no ladder breaks.",
    },
    {
      agent: "Features",
      title: "We engineered a clean feature set",
      description:
        "Lagged prices, seasonality, promo flags, and competitor signals — all checked for collinearity (VIF < 10).",
      implication: "Every input earns its keep before modelling.",
    },
    {
      agent: "Modeling",
      title: "We fit price-elasticity models",
      description:
        "Log-log regression with rolling-origin cross-validation. Sign and magnitude are sanity-checked against history.",
      implication: "We know how each PPG responds to price.",
    },
    {
      agent: "Decomposition",
      title: "We attributed every unit to a driver",
      description:
        "Each week's volume is split into base, price, promo, seasonality, and shock — residuals stay under 1%.",
      implication: "Nothing unexplained hides in the recommendation.",
    },
    {
      agent: "Optimisation",
      title: "We searched the constrained price space",
      description:
        "Mixed-integer programme respecting ladder, margin floor, and competitor-gap rules. No surprises.",
      implication: "Every proposed price is feasible by construction.",
    },
    {
      agent: "Validation",
      title: "We trust-checked the answer",
      description:
        "Hold-out WAPE, sign checks, constraint feasibility, and confidence bands before anything reaches you.",
      implication: "Recommendations clear six gates before you see them.",
    },
  ];
}

export function buildMockPayload(runId: string | null = null): ClientPayload {
  const recommendations = buildRecommendations();
  const weekly = buildWeekly(recommendations);
  const totalLift = recommendations.reduce((a, r) => a + r.revenue_lift_usd, 0);
  const repriced = recommendations.filter((r) => Math.abs(r.delta_pct) > 0.005).length;
  const avgConfidence =
    recommendations.reduce((a, r) => a + r.confidence, 0) / recommendations.length;
  const flagged = recommendations.filter((r) => r.flagged).length;

  const liftPct = (totalLift / 5_400_000) * 100;
  const anchor = {
    label: "Forecast quarterly revenue lift",
    value: `+$${(totalLift / 1000).toFixed(0)}K`,
    delta: `+${liftPct.toFixed(1)}%`,
    detail: "13-week projection vs. status-quo pricing.",
  };

  const kpis: Kpi[] = [
    {
      label: "PPGs re-priced",
      value: `${repriced} / ${recommendations.length}`,
      hint: "4 up-ticks · 4 down-ticks",
      positive: true,
    },
    {
      label: "Model confidence",
      value: `${(avgConfidence * 100).toFixed(0)}%`,
      hint: "Lowest 82% · highest 94%",
      positive: true,
    },
    {
      label: "For review",
      value: `${flagged}`,
      hint: flagged
        ? "Below 84% confidence — eyeball before commit"
        : "All cleared the trust threshold",
      positive: flagged === 0,
    },
  ];

  const validation = buildValidation();
  const passes = validation.filter((v) => v.status === "pass").length;
  const trust_score = Math.round((passes / validation.length) * 100);

  return {
    run_id: runId,
    generated_at: new Date().toISOString(),
    anchor,
    kpis,
    weekly,
    recommendations,
    elasticity_by_ppg: buildElasticityCurves(recommendations),
    validation,
    trust_score,
    forest: buildForest(recommendations),
    methodology: buildMethodology(),
    narrative:
      "We recommend repricing 6 of 8 Price-Pack Groups: small up-ticks on inelastic premium packs and gentle EDLP cuts on highly elastic multipacks. Every proposed price clears your margin floor and ladder rules. One sits at the competitor-gap ceiling and is flagged for review.",
  };
}
