// Mock data drawn from a real completed run on the synthetic panel.
// Used by every mockup layout so comparisons are apples-to-apples.

export const TREND = Array.from({ length: 26 }, (_, i) => {
  const week = `W${String(i + 1).padStart(2, "0")}`;
  const base = 1850 + Math.sin(i / 4) * 350 + (i > 18 ? 320 : 0);
  const promo = i % 5 === 0 ? 1 : 0;
  return {
    week,
    units: Math.round(base + (promo ? 480 : 0) + (Math.random() - 0.5) * 90),
    price: +(2.5 - (promo ? 0.55 : 0) + Math.cos(i / 5) * 0.06).toFixed(2),
    promo,
  };
});

export const PPGS = [
  { id: "PPG_AUTO_01", brand: "AlphaCola", category: "soda", colour: "#34d399" },
  { id: "PPG_AUTO_02", brand: "BetaCola", category: "soda", colour: "#60a5fa" },
  { id: "PPG_AUTO_03", brand: "Spark", category: "energy", colour: "#f472b6" },
  { id: "PPG_AUTO_04", brand: "Zest", category: "energy", colour: "#fbbf24" },
  { id: "PPG_AUTO_05", brand: "Hydro", category: "water", colour: "#22d3ee" },
  { id: "PPG_AUTO_06", brand: "Pure", category: "water", colour: "#a78bfa" },
  { id: "PPG_AUTO_07", brand: "Brew", category: "tea", colour: "#fb7185" },
  { id: "PPG_AUTO_08", brand: "Leaf", category: "tea", colour: "#4ade80" },
];

// 48 SKUs, 6 per PPG: small / medium / large × 2 variants
export const SKU_SCATTER = PPGS.flatMap((p, pi) =>
  Array.from({ length: 6 }, (_, i) => {
    const pack = i < 2 ? "small" : i < 4 ? "medium" : "large";
    const tier = pack === "small" ? 0 : pack === "medium" ? 1 : 2;
    const basePrice = 1.5 + tier * 0.65 + pi * 0.04;
    return {
      sku: `${p.id.slice(-2)}-${String(i + 1).padStart(2, "0")}`,
      ppg: p.id,
      brand: p.brand,
      pack,
      x: tier + (Math.random() - 0.5) * 0.18,
      y: +(Math.log(basePrice) + (Math.random() - 0.5) * 0.08).toFixed(3),
      colour: p.colour,
    };
  }),
);

export const PPG_PRICE_BOX = PPGS.map((p, pi) => {
  const c = 1.5 + pi * 0.12;
  return {
    ppg: p.id.slice(-2),
    min: +(c).toFixed(2),
    q1: +(c + 0.35).toFixed(2),
    median: +(c + 0.95).toFixed(2),
    q3: +(c + 1.55).toFixed(2),
    max: +(c + 2.1).toFixed(2),
    colour: p.colour,
  };
});

export const VIF = [
  { feature: "log_price", vif: 7.92 },
  { feature: "lag1_log_units", vif: 7.38 },
  { feature: "tpr_share", vif: 4.55 },
  { feature: "discount_depth", vif: 3.81 },
  { feature: "log_price_gap", vif: 1.83 },
  { feature: "display_share", vif: 1.5 },
  { feature: "feature_share", vif: 1.36 },
  { feature: "log_distribution_acv", vif: 1.3 },
  { feature: "week_sin", vif: 1.12 },
  { feature: "is_holiday_week", vif: 1.05 },
  { feature: "week_cos", vif: 1.03 },
];

export const DROPPED = [
  { feature: "log_base_price", reason: "|corr|=1.00 with log_competitor_price" },
  { feature: "log_competitor_price", reason: "|corr|=0.99 with log_price" },
  { feature: "lag4_log_price", reason: "|corr|=0.99 with log_price" },
  { feature: "lag1_log_price", reason: "|corr|=0.99 with log_price" },
];

const CORR_FEATURES = [
  "log_price",
  "discount_depth",
  "tpr_share",
  "display_share",
  "feature_share",
  "log_distribution_acv",
  "log_price_gap",
  "lag1_log_units",
  "week_sin",
  "week_cos",
  "is_holiday_week",
];

function seededCorr(): number[][] {
  // hand-shaped matrix that reads like a real refined feature set
  const n = CORR_FEATURES.length;
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    m[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) m[i][j] = 1;
      else {
        const base = Math.sin((i + 1) * (j + 1) * 0.7) * 0.55;
        m[i][j] = +Math.max(-0.92, Math.min(0.92, base)).toFixed(2);
      }
    }
  }
  // Symmetrise
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) m[j][i] = m[i][j];
  return m;
}

export const CORR_LABELS = CORR_FEATURES;
export const CORR_MATRIX = seededCorr();

export const PREVIEW_ROWS = [
  { sku: "PPG01-01", week: "2024-09-02", store: "ST-01", units: 412, price: 1.49, base_price: 1.59, tpr: 1 },
  { sku: "PPG01-01", week: "2024-09-09", store: "ST-01", units: 318, price: 1.59, base_price: 1.59, tpr: 0 },
  { sku: "PPG01-02", week: "2024-09-09", store: "ST-02", units: 256, price: 1.99, base_price: 1.99, tpr: 0 },
  { sku: "PPG03-04", week: "2024-09-09", store: "ST-01", units: 89, price: 3.49, base_price: 3.79, tpr: 1 },
  { sku: "PPG05-02", week: "2024-09-09", store: "ST-03", units: 1041, price: 0.99, base_price: 1.19, tpr: 1 },
  { sku: "PPG07-03", week: "2024-09-16", store: "ST-04", units: 482, price: 2.79, base_price: 2.99, tpr: 1 },
  { sku: "PPG08-01", week: "2024-09-16", store: "ST-05", units: 198, price: 2.49, base_price: 2.49, tpr: 0 },
];

export const SCHEMA = [
  { column: "sku", dtype: "VARCHAR", role: "identifier", nulls: "0.0%" },
  { column: "week_start", dtype: "DATE", role: "temporal", nulls: "0.0%" },
  { column: "store_id", dtype: "VARCHAR", role: "identifier", nulls: "0.0%" },
  { column: "ppg_id", dtype: "VARCHAR", role: "identifier", nulls: "0.0%" },
  { column: "units", dtype: "INTEGER", role: "target", nulls: "0.0%" },
  { column: "price", dtype: "DOUBLE", role: "numeric", nulls: "0.0%" },
  { column: "base_price", dtype: "DOUBLE", role: "numeric", nulls: "0.0%" },
  { column: "discount_depth", dtype: "DOUBLE", role: "numeric", nulls: "0.0%" },
  { column: "tpr_flag", dtype: "INTEGER", role: "flag", nulls: "0.0%" },
  { column: "competitor_price", dtype: "DOUBLE", role: "numeric", nulls: "0.0%" },
  { column: "distribution_acv", dtype: "DOUBLE", role: "numeric", nulls: "0.0%" },
  { column: "holiday", dtype: "VARCHAR", role: "temporal", nulls: "92.3%" },
];

export const QUALITY_CHECKS = [
  { source: "dbt", name: "unique_panel_keys", status: "pass", severity: "error", message: "48 SKUs × 5 stores × 104 weeks unique" },
  { source: "dbt", name: "price_positive", status: "pass", severity: "error", message: "" },
  { source: "dbt", name: "units_nonneg", status: "pass", severity: "error", message: "" },
  { source: "dbt", name: "tpr_implies_discount", status: "pass", severity: "error", message: "" },
  { source: "dbt", name: "weekly_continuity", status: "pass", severity: "error", message: "" },
  { source: "dbt", name: "discount_depth_in_-1_to_0.5", status: "warn", severity: "warn", message: "2912 rows above 0.5 — investigate deep-cut promos" },
  { source: "ge", name: "expect_column_values_to_be_between(price, 0.5, 10)", status: "pass", severity: "error", message: "" },
  { source: "ge", name: "expect_column_pair_values_a_to_be_geq_b(base_price, price)", status: "pass", severity: "error", message: "" },
  { source: "ge", name: "expect_table_row_count_to_be_between", status: "pass", severity: "error", message: "24960 / expected 12000–30000" },
  { source: "ge", name: "expect_column_mean_to_be_between(units, 50, 500)", status: "warn", severity: "warn", message: "mean=412.7 within band but elevated tail" },
];

export const ANOMALIES = [
  { sku: "PPG05-02", week: "2024-12-23", units: 4108, price: 0.79, reason: "units z=+4.2 (vs PPG mean) — holiday TPR" },
  { sku: "PPG03-04", week: "2024-07-01", units: 12, price: 3.49, reason: "units z=-3.6 — likely OOS" },
  { sku: "PPG02-05", week: "2024-04-15", units: 902, price: 2.10, reason: "price IQR upper outlier" },
  { sku: "PPG06-01", week: "2024-08-19", units: 88, price: 1.49, reason: "discount_depth=0.61 (>0.5 floor)" },
  { sku: "PPG07-03", week: "2024-11-04", units: 1284, price: 2.59, reason: "tpr=1 but discount_depth=0.03 — config mismatch" },
];

export const COVERAGE = (() => {
  const skus = SKU_SCATTER.map((r) => r.sku).slice(0, 24);
  const weeks = 26;
  return skus.map((sku, i) => ({
    sku,
    cells: Array.from({ length: weeks }, (_, w) => {
      const dropout = (i === 4 && w > 18) || (i === 11 && w === 7) || (i === 19 && w === 21);
      return dropout ? 0 : 1;
    }),
  }));
})();
