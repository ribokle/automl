"""Pure stats helpers behind the ``advanced_eda`` agent.

Every function returns a JSON-serialisable shape so the agent can drop the
result straight into a run artefact. No DuckDB, no LLM, no side effects — the
agent owns IO and orchestration.

Compute grain decisions (from the design review):

* Time-series diagnostics (STL, ACF, ADF, isolation forest) run on the
  **PPG × week** roll-up, capped at the top-K PPGs by revenue. Smaller PPGs
  get cheap summaries only.
* Cross-PPG correlation is computed **within-category**, capped at the top-20
  PPGs per category. Avoids the O(N²) blow-up on Dominick's.
* The univariate log-log slope is named ``price_volume_slope`` — never
  ``elasticity`` — and carries an explicit caveat field so the operator
  doesn't confuse it with the modelling agent's controlled estimate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_float(x: float | int | np.floating, default: float = 0.0) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def top_k_ppgs_by_revenue(panel: pd.DataFrame, k: int) -> list[str]:
    """Return the K PPGs with the highest cumulative units*price spend."""
    if panel.empty or "ppg_id" not in panel.columns:
        return []
    revenue = (
        panel.assign(rev=panel["units"].fillna(0) * panel["price"].fillna(0))
        .groupby("ppg_id")["rev"]
        .sum()
        .sort_values(ascending=False)
    )
    return revenue.head(k).index.tolist()


def store_variability(raw_panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Per PPG cross-store variability sidecar — CV of weekly units and
    share of zero-unit weeks. Cheap enough to compute for every PPG."""
    if raw_panel.empty:
        return []
    needed = {"ppg_id", "store_id", "week_start", "units"}
    if not needed.issubset(raw_panel.columns):
        return []
    weekly = (
        raw_panel.groupby(["ppg_id", "store_id", "week_start"], as_index=False)["units"]
        .sum()
    )
    out: list[dict[str, Any]] = []
    for ppg, grp in weekly.groupby("ppg_id"):
        store_stats = grp.groupby("store_id")["units"].agg(["mean", "std"]).reset_index()
        store_stats["cv"] = store_stats["std"] / store_stats["mean"].replace(0, np.nan)
        zero_share = float((grp["units"] == 0).mean())
        out.append(
            {
                "ppg_id": ppg,
                "n_stores": int(grp["store_id"].nunique()),
                "store_cv_median": _safe_float(store_stats["cv"].median()),
                "store_cv_max": _safe_float(store_stats["cv"].max()),
                "zero_week_share": zero_share,
            }
        )
    return out


def stl_decomposition(series: pd.Series, period: int = 52) -> dict[str, Any]:
    """STL decomposition of a weekly series. Period defaults to 52 (annual
    seasonality). Returns observed / trend / seasonal / residual arrays."""
    if series is None or len(series.dropna()) < period * 2:
        return {
            "available": False,
            "reason": f"need >= {period * 2} non-null points",
            "observed": [], "trend": [], "seasonal": [], "resid": [], "weeks": [],
        }
    from statsmodels.tsa.seasonal import STL

    s = series.copy().asfreq("W", method="ffill") if isinstance(series.index, pd.DatetimeIndex) else series.copy()
    s = s.interpolate(method="linear").bfill().ffill()
    try:
        res = STL(s, period=period, robust=True).fit()
    except Exception:  # noqa: BLE001
        return {
            "available": False, "reason": "STL fit failed",
            "observed": [], "trend": [], "seasonal": [], "resid": [], "weeks": [],
        }
    idx = s.index
    weeks = [w.isoformat() if hasattr(w, "isoformat") else str(w) for w in idx]
    seasonal_amplitude = _safe_float(res.seasonal.max() - res.seasonal.min())
    trend_range = _safe_float(res.trend.max() - res.trend.min())
    return {
        "available": True,
        "weeks": weeks,
        "observed": [round(_safe_float(v), 3) for v in s.values],
        "trend": [round(_safe_float(v), 3) for v in res.trend.values],
        "seasonal": [round(_safe_float(v), 3) for v in res.seasonal.values],
        "resid": [round(_safe_float(v), 3) for v in res.resid.values],
        "seasonal_amplitude": round(seasonal_amplitude, 3),
        "trend_range": round(trend_range, 3),
    }


def acf_pacf(series: pd.Series, nlags: int = 24) -> dict[str, Any]:
    """ACF + PACF up to ``nlags`` with a 95% confidence band."""
    from statsmodels.tsa.stattools import acf, pacf

    s = series.dropna().astype(float)
    if len(s) < nlags + 2:
        return {"available": False, "reason": "series too short", "acf": [], "pacf": [], "lags": [], "ci": 0.0}
    try:
        ac = acf(s.values, nlags=nlags, fft=True)
        pc = pacf(s.values, nlags=nlags, method="yw")
    except Exception:  # noqa: BLE001
        return {"available": False, "reason": "stat fit failed", "acf": [], "pacf": [], "lags": [], "ci": 0.0}
    ci = 1.96 / math.sqrt(len(s))
    return {
        "available": True,
        "lags": list(range(nlags + 1)),
        "acf": [round(_safe_float(v), 4) for v in ac],
        "pacf": [round(_safe_float(v), 4) for v in pc],
        "ci": round(ci, 4),
    }


def stationarity_tests(series: pd.Series) -> dict[str, Any]:
    """ADF + KPSS. Verdict combines both:
    - stationary if ADF rejects AND KPSS fails to reject
    - non-stationary if ADF fails to reject AND KPSS rejects
    - inconclusive otherwise."""
    from statsmodels.tsa.stattools import adfuller, kpss

    s = series.dropna().astype(float)
    if len(s) < 20 or s.std() < 1e-9:
        return {"available": False, "adf_p": None, "kpss_p": None, "verdict": "insufficient"}
    try:
        adf_p = float(adfuller(s.values, autolag="AIC")[1])
    except Exception:  # noqa: BLE001
        adf_p = None
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = float(kpss(s.values, regression="c", nlags="auto")[1])
    except Exception:  # noqa: BLE001
        kpss_p = None
    verdict = "inconclusive"
    if adf_p is not None and kpss_p is not None:
        adf_reject = adf_p < 0.05
        kpss_reject = kpss_p < 0.05
        if adf_reject and not kpss_reject:
            verdict = "stationary"
        elif (not adf_reject) and kpss_reject:
            verdict = "non_stationary"
    return {
        "available": True,
        "adf_p": _safe_float(adf_p) if adf_p is not None else None,
        "kpss_p": _safe_float(kpss_p) if kpss_p is not None else None,
        "verdict": verdict,
    }


def time_series_diagnostics(
    panel: pd.DataFrame, ppgs: list[str]
) -> list[dict[str, Any]]:
    """Per-PPG bundle: STL + ACF/PACF + stationarity."""
    out: list[dict[str, Any]] = []
    for ppg in ppgs:
        sub = panel[panel["ppg_id"] == ppg].sort_values("week_start")
        if sub.empty:
            continue
        series = pd.Series(
            sub["units"].astype(float).values,
            index=pd.to_datetime(sub["week_start"]),
            name=ppg,
        )
        out.append(
            {
                "ppg_id": ppg,
                "n_weeks": int(len(series)),
                "stl": stl_decomposition(series),
                "acf_pacf": acf_pacf(series),
                "stationarity": stationarity_tests(series),
            }
        )
    return out


def detect_stockouts(panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows where ``units == 0`` with ``distribution_acv > 0`` and the prior
    week had positive units at unchanged price. Heuristic — picks up the
    classic out-of-stock signature without being noisy."""
    if panel.empty:
        return []
    df = panel.copy().sort_values(["ppg_id", "week_start"])
    df["prev_units"] = df.groupby("ppg_id")["units"].shift(1)
    df["prev_price"] = df.groupby("ppg_id")["price"].shift(1)
    flags = (
        (df["units"] == 0)
        & (df["distribution_acv"].fillna(100) > 0)
        & (df["prev_units"].fillna(0) > 0)
        & (np.abs(df["price"].fillna(0) - df["prev_price"].fillna(0)) <= 0.01)
    )
    out = []
    for r in df[flags].itertuples(index=False):
        out.append(
            {
                "ppg_id": r.ppg_id,
                "week_start": r.week_start.isoformat() if hasattr(r.week_start, "isoformat") else str(r.week_start),
                "anomaly_type": "stockout",
                "severity": float(min(1.0, _safe_float(r.prev_units) / max(1.0, _safe_float(r.prev_units)))),
                "note": f"prev wk units={int(_safe_float(r.prev_units))} at price=${_safe_float(r.prev_price):.2f}",
            }
        )
    return out


def detect_pantry_loading(panel: pd.DataFrame, lookback: int = 4) -> list[dict[str, Any]]:
    """A spike > 2σ above PPG mean followed by a trough < −1σ within
    ``lookback`` weeks — the canonical post-promo pull-forward shape."""
    if panel.empty:
        return []
    df = panel.copy().sort_values(["ppg_id", "week_start"])
    out: list[dict[str, Any]] = []
    for ppg, grp in df.groupby("ppg_id"):
        u = grp["units"].astype(float)
        if u.std() < 1e-6:
            continue
        mean = u.mean()
        std = u.std()
        z = (u - mean) / std
        z_arr = z.values
        weeks = grp["week_start"].tolist()
        n = len(z_arr)
        for i in range(n):
            if z_arr[i] <= 2.0:
                continue
            for j in range(i + 1, min(n, i + 1 + lookback)):
                if z_arr[j] < -1.0:
                    out.append(
                        {
                            "ppg_id": ppg,
                            "week_start": weeks[i].isoformat() if hasattr(weeks[i], "isoformat") else str(weeks[i]),
                            "anomaly_type": "pantry_loading",
                            "severity": round(float(z_arr[i]), 3),
                            "note": f"spike z={z_arr[i]:.1f}σ at +0, trough z={z_arr[j]:.1f}σ at +{j - i}wk",
                        }
                    )
                    break
    return out


def detect_forward_buy(panel: pd.DataFrame, lookback: int = 4) -> list[dict[str, Any]]:
    """Units spike > 1.5σ in the ``lookback`` window preceding a base-price
    increase ≥ 5%."""
    if panel.empty:
        return []
    df = panel.copy().sort_values(["ppg_id", "week_start"])
    out: list[dict[str, Any]] = []
    for ppg, grp in df.groupby("ppg_id"):
        u = grp["units"].astype(float).values
        bp = grp["base_price"].astype(float).values
        weeks = grp["week_start"].tolist()
        n = len(u)
        if n < lookback + 2:
            continue
        std = u.std()
        mean = u.mean()
        if std < 1e-6:
            continue
        for i in range(lookback, n):
            prev_bp = bp[i - 1]
            if prev_bp <= 0 or np.isnan(prev_bp):
                continue
            pct = (bp[i] - prev_bp) / prev_bp
            if pct < 0.05:
                continue
            window = u[max(0, i - lookback):i]
            window_max = window.max()
            z = (window_max - mean) / std
            if z > 1.5:
                spike_offset = int(lookback - (np.argmax(window) + 1))
                out.append(
                    {
                        "ppg_id": ppg,
                        "week_start": weeks[i].isoformat() if hasattr(weeks[i], "isoformat") else str(weeks[i]),
                        "anomaly_type": "forward_buy",
                        "severity": round(float(z), 3),
                        "note": f"spike z={z:.1f}σ {spike_offset}wk before base-price +{pct * 100:.1f}%",
                    }
                )
    return out


def isolation_forest_anomalies(panel: pd.DataFrame, ppgs: list[str]) -> list[dict[str, Any]]:
    """Per-PPG isolation forest on (log_price, discount_depth, lag1_log_units).
    Returns rows with anomaly score > 0.6 (rescaled from -score). Skips PPGs
    with too few observations."""
    from sklearn.ensemble import IsolationForest

    if panel.empty:
        return []
    out: list[dict[str, Any]] = []
    for ppg in ppgs:
        sub = panel[panel["ppg_id"] == ppg].sort_values("week_start").copy()
        if len(sub) < 20:
            continue
        sub["log_price"] = np.log(sub["price"].clip(lower=1e-6))
        sub["log_units"] = np.log(sub["units"].clip(lower=1))
        sub["lag1_log_units"] = sub["log_units"].shift(1)
        features = sub[["log_price", "discount_depth", "lag1_log_units"]].dropna()
        if len(features) < 20:
            continue
        forest = IsolationForest(
            n_estimators=80, contamination="auto", random_state=42, n_jobs=1
        )
        forest.fit(features.values)
        scores = -forest.score_samples(features.values)
        rescaled = (scores - scores.min()) / max(1e-9, scores.max() - scores.min())
        for idx, score in zip(features.index, rescaled, strict=False):
            if score < 0.6:
                continue
            row = sub.loc[idx]
            wk = row["week_start"]
            out.append(
                {
                    "ppg_id": ppg,
                    "week_start": wk.isoformat() if hasattr(wk, "isoformat") else str(wk),
                    "anomaly_type": "isolation_forest",
                    "severity": round(float(score), 3),
                    "note": f"feature-space outlier (log_price={row['log_price']:.2f}, depth={row['discount_depth']:.2f})",
                }
            )
    return out


def detect_change_points(panel: pd.DataFrame, ppgs: list[str], min_size: int = 10) -> list[dict[str, Any]]:
    """PELT change-point detection on baseline price per PPG. Flags
    structural shifts the modelling agent shouldn't try to explain as price
    response."""
    import ruptures as rpt

    out: list[dict[str, Any]] = []
    for ppg in ppgs:
        sub = panel[panel["ppg_id"] == ppg].sort_values("week_start")
        if len(sub) < min_size * 2:
            continue
        bp = sub["base_price"].astype(float).ffill().values
        if np.std(bp) < 1e-6:
            continue
        try:
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(bp.reshape(-1, 1))
            breaks = algo.predict(pen=max(1.0, np.var(bp) * 4.0))
        except Exception:  # noqa: BLE001
            continue
        weeks = sub["week_start"].tolist()
        for b in breaks[:-1]:
            if b <= 0 or b >= len(weeks):
                continue
            wk = weeks[b]
            pre = float(np.mean(bp[max(0, b - min_size):b]))
            post = float(np.mean(bp[b:b + min_size]))
            out.append(
                {
                    "ppg_id": ppg,
                    "week_start": wk.isoformat() if hasattr(wk, "isoformat") else str(wk),
                    "pre_base_price": round(pre, 3),
                    "post_base_price": round(post, 3),
                    "delta_pct": round((post - pre) / max(1e-6, pre) * 100, 2),
                }
            )
    return out


def distribution_report(panel: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Per numeric column: skew, kurtosis, Shapiro p-value (sampled at 5000
    if larger)."""
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=0)
    for c in columns:
        if c not in panel.columns:
            continue
        s = pd.to_numeric(panel[c], errors="coerce").dropna()
        if s.empty:
            continue
        n = int(s.size)
        sample = s.values if n <= 5000 else rng.choice(s.values, size=5000, replace=False)
        try:
            shapiro_p = float(stats.shapiro(sample).pvalue) if len(sample) >= 3 else None
        except Exception:  # noqa: BLE001
            shapiro_p = None
        out.append(
            {
                "column": c,
                "n": n,
                "skew": round(_safe_float(stats.skew(s.values)), 4),
                "kurtosis": round(_safe_float(stats.kurtosis(s.values)), 4),
                "shapiro_p": _safe_float(shapiro_p) if shapiro_p is not None else None,
                "iqr": round(_safe_float(s.quantile(0.75) - s.quantile(0.25)), 4),
            }
        )
    return out


def _bootstrap_mean_ci(
    a: np.ndarray, b: np.ndarray, n_iter: int = 200, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """Bootstrap mean(a)/mean(b) and a 95% CI. Returns (point, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(seed=0)
    if len(a) == 0 or len(b) == 0:
        return (0.0, 0.0, 0.0)
    base_a = a.mean()
    base_b = b.mean()
    if base_b <= 0:
        return (0.0, 0.0, 0.0)
    point = base_a / base_b
    samples = []
    for _ in range(n_iter):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        mb = b_s.mean()
        samples.append(a_s.mean() / mb if mb > 0 else 0.0)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return (float(point), float(lo), float(hi))


def promo_lift_sketches(panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-PPG pre-model lift sketches for TPR, display, feature flags, plus
    promo-window length distribution.

    Lift = mean(units when flag=1) / mean(units when flag=0). The
    modelling agent recovers the controlled estimate; this is a pre-model
    sniff for sanity-checking the assortment."""
    if panel.empty:
        return []
    rng = np.random.default_rng(seed=0)
    out: list[dict[str, Any]] = []
    for ppg, grp in panel.groupby("ppg_id"):
        u = grp["units"].astype(float).values
        tpr = grp.get("tpr_share", grp.get("tpr_flag", pd.Series(dtype=float)))
        dsp = grp.get("display_share", grp.get("display_flag", pd.Series(dtype=float)))
        feat = grp.get("feature_share", grp.get("feature_flag", pd.Series(dtype=float)))
        results: dict[str, dict[str, float]] = {}
        for flag_name, flag in (("tpr", tpr), ("display", dsp), ("feature", feat)):
            if flag is None or len(flag) == 0:
                continue
            vals = flag.fillna(0).values.astype(float)
            # Median-split per PPG: store-level promo shares rarely sync above
            # 0.5, so an absolute threshold leaves the "on" bucket empty.
            cutoff = float(np.median(vals[vals > 0])) if (vals > 0).any() else 0.0
            mask = vals >= max(cutoff, 1e-6)
            on, off = u[mask], u[~mask]
            if len(on) < 3 or len(off) < 3:
                results[flag_name] = {"lift": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_on": int(len(on))}
                continue
            lift, lo, hi = _bootstrap_mean_ci(on, off, rng=rng)
            results[flag_name] = {
                "lift": round(lift, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                "n_on": int(len(on)),
            }
        if tpr is not None and len(tpr) > 0:
            tpr_vals = tpr.fillna(0).values.astype(float)
            tpr_cutoff = float(np.median(tpr_vals[tpr_vals > 0])) if (tpr_vals > 0).any() else 0.0
            tpr_mask = tpr_vals >= max(tpr_cutoff, 1e-6)
        else:
            tpr_mask = np.zeros(len(grp), dtype=bool)
        runs = []
        cur = 0
        for flag in tpr_mask:
            if flag:
                cur += 1
            elif cur > 0:
                runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        window_stats = (
            {
                "modal_length": int(stats.mode(runs, keepdims=False).mode) if runs else 0,
                "mean_length": round(float(np.mean(runs)), 2) if runs else 0.0,
                "max_length": int(max(runs)) if runs else 0,
                "n_windows": len(runs),
                "pct_multiweek": round(float(np.mean([r > 1 for r in runs])), 3) if runs else 0.0,
            }
        )
        out.append({"ppg_id": ppg, "lifts": results, "promo_window": window_stats})
    return out


def cross_ppg_correlation(
    panel: pd.DataFrame, ppg_categories: dict[str, str], per_cat_cap: int = 20
) -> dict[str, Any]:
    """Within-category correlation of weekly units across PPGs.

    Detrended via week-over-week residual against the per-PPG mean. Capped
    at ``per_cat_cap`` PPGs per category by revenue. Output is one heatmap
    per category. The name is *correlation*, not *cannibalisation* — at this
    stage we can't separate substitution from common-cause drivers."""
    if panel.empty:
        return {"categories": []}
    cats: dict[str, list[str]] = {}
    for ppg_id, cat in ppg_categories.items():
        cats.setdefault(cat or "uncategorised", []).append(ppg_id)
    revenue = (
        panel.assign(rev=panel["units"].fillna(0) * panel["price"].fillna(0))
        .groupby("ppg_id")["rev"]
        .sum()
    )
    cat_blobs: list[dict[str, Any]] = []
    for cat, ppgs in cats.items():
        ranked = sorted(ppgs, key=lambda p: revenue.get(p, 0.0), reverse=True)[:per_cat_cap]
        if len(ranked) < 2:
            continue
        wide = (
            panel[panel["ppg_id"].isin(ranked)]
            .pivot_table(index="week_start", columns="ppg_id", values="units", aggfunc="sum")
        )
        if wide.shape[1] < 2 or len(wide) < 8:
            continue
        detrended = wide.sub(wide.mean(axis=0), axis=1)
        corr = detrended.corr()
        labels = list(corr.columns)
        cat_blobs.append(
            {
                "category": cat,
                "labels": labels,
                "matrix": [[round(_safe_float(corr.loc[a, b]), 3) for b in labels] for a in labels],
            }
        )
    return {"categories": cat_blobs}


def pareto_abc(
    raw_panel: pd.DataFrame, ppg_units: pd.DataFrame
) -> dict[str, Any]:
    """Pareto curves for SKU / brand / store on both volume and revenue, plus
    ABC class per PPG (80 / 15 / 5 cumulative-revenue bands)."""
    out: dict[str, Any] = {"sku": {}, "brand": {}, "store": {}, "ppg_abc": []}

    def _lorenz(values: pd.Series) -> dict[str, Any]:
        v = values.sort_values(ascending=False).values
        if len(v) == 0:
            return {"x": [], "cum_share": [], "labels": []}
        total = float(v.sum())
        if total <= 0:
            return {"x": [], "cum_share": [], "labels": []}
        cum = np.cumsum(v) / total
        return {
            "x": [round(float(i + 1) / len(v), 4) for i in range(len(v))],
            "cum_share": [round(float(c), 4) for c in cum],
            "labels": [str(idx) for idx in values.sort_values(ascending=False).index.tolist()],
        }

    if "sku" in raw_panel.columns:
        sku_units = raw_panel.groupby("sku")["units"].sum()
        sku_rev = (raw_panel.assign(rev=raw_panel["units"] * raw_panel["price"]).groupby("sku")["rev"].sum())
        out["sku"] = {"volume": _lorenz(sku_units), "revenue": _lorenz(sku_rev)}
    if "brand" in raw_panel.columns:
        brand_units = raw_panel.groupby("brand")["units"].sum()
        brand_rev = (raw_panel.assign(rev=raw_panel["units"] * raw_panel["price"]).groupby("brand")["rev"].sum())
        out["brand"] = {"volume": _lorenz(brand_units), "revenue": _lorenz(brand_rev)}
    if "store_id" in raw_panel.columns:
        store_units = raw_panel.groupby("store_id")["units"].sum()
        store_rev = (raw_panel.assign(rev=raw_panel["units"] * raw_panel["price"]).groupby("store_id")["rev"].sum())
        out["store"] = {"volume": _lorenz(store_units), "revenue": _lorenz(store_rev)}

    if not ppg_units.empty and "ppg_id" in ppg_units.columns:
        ppg_rev = (
            ppg_units.assign(rev=ppg_units["units"].fillna(0) * ppg_units["price"].fillna(0))
            .groupby("ppg_id")["rev"]
            .sum()
            .sort_values(ascending=False)
        )
        total = float(ppg_rev.sum())
        cum_dollars = 0.0
        for ppg, rev in ppg_rev.items():
            share_prior = (cum_dollars / total) if total > 0 else 0.0
            cum_dollars += float(rev)
            share_cum = (cum_dollars / total) if total > 0 else 0.0
            # Classify by where this item *starts* on the cumulative curve so
            # the item that single-handedly crosses 80% still lands in A.
            if share_prior < 0.80:
                klass = "A"
            elif share_prior < 0.95:
                klass = "B"
            else:
                klass = "C"
            out["ppg_abc"].append(
                {
                    "ppg_id": ppg,
                    "revenue": round(float(rev), 2),
                    "cum_share": round(float(share_cum), 4),
                    "abc_class": klass,
                }
            )
    return out


def price_ladder(panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Per PPG: distinct price points + frequency, plus a univariate log-log
    OLS slope (named ``price_volume_slope``, NOT elasticity)."""
    if panel.empty:
        return []
    out: list[dict[str, Any]] = []
    for ppg, grp in panel.groupby("ppg_id"):
        prices = grp["price"].astype(float).round(2)
        units = grp["units"].astype(float)
        ladder = (
            prices.value_counts()
            .sort_index()
            .rename_axis("price")
            .reset_index(name="n_weeks")
            .sort_values("price")
        )
        mask = (prices > 0) & (units > 0)
        slope: float | None = None
        intercept: float | None = None
        if mask.sum() >= 10:
            lp = np.log(prices[mask].values)
            lu = np.log(units[mask].values)
            if lp.std() > 1e-6:
                try:
                    slope_, intercept_ = np.polyfit(lp, lu, 1)
                    slope = round(float(slope_), 3)
                    intercept = round(float(intercept_), 3)
                except Exception:  # noqa: BLE001
                    pass
        out.append(
            {
                "ppg_id": ppg,
                "n_distinct_prices": int(len(ladder)),
                "ladder": [
                    {"price": round(float(r.price), 2), "n_weeks": int(r.n_weeks)}
                    for r in ladder.itertuples(index=False)
                ],
                "price_volume_slope": slope,
                "log_intercept": intercept,
                "caveat": "univariate log-log slope; not causal — see modeling agent for controlled estimate",
            }
        )
    return out


def promo_calendar(panel: pd.DataFrame) -> dict[str, Any]:
    """Week × PPG matrix where each cell is the dominant promo type for
    that PPG-week (none / tpr / display / feature / multi)."""
    if panel.empty:
        return {"weeks": [], "ppgs": [], "matrix": []}
    df = panel.copy()
    weeks = sorted(df["week_start"].unique().tolist())
    ppgs = sorted(df["ppg_id"].unique().tolist())
    wk_idx = {w: i for i, w in enumerate(weeks)}
    pp_idx = {p: i for i, p in enumerate(ppgs)}
    matrix = [["none"] * len(weeks) for _ in ppgs]
    # Threshold of 0.1 fires whenever a meaningful share of stores ran the
    # promo type that week; for store-level all-or-nothing flags this is
    # equivalent to "any promo".
    for r in df.itertuples(index=False):
        flags = []
        if getattr(r, "tpr_share", 0) and r.tpr_share >= 0.1:
            flags.append("tpr")
        if getattr(r, "display_share", 0) and r.display_share >= 0.1:
            flags.append("display")
        if getattr(r, "feature_share", 0) and r.feature_share >= 0.1:
            flags.append("feature")
        if len(flags) == 0:
            cell = "none"
        elif len(flags) == 1:
            cell = flags[0]
        else:
            cell = "multi"
        i = pp_idx[r.ppg_id]
        j = wk_idx[r.week_start]
        matrix[i][j] = cell
    return {
        "weeks": [w.isoformat() if hasattr(w, "isoformat") else str(w) for w in weeks],
        "ppgs": ppgs,
        "matrix": matrix,
    }


def holiday_lift_table(raw_panel: pd.DataFrame) -> list[dict[str, Any]]:
    """For each holiday week, per-PPG units delta vs trailing 4-week median."""
    if raw_panel.empty or "holiday" not in raw_panel.columns:
        return []
    df = raw_panel.copy()
    df["holiday"] = df["holiday"].fillna("")
    weekly = df.groupby(["ppg_id", "week_start"], as_index=False).agg(
        units=("units", "sum"),
        holiday=("holiday", lambda x: next((h for h in x if h), "")),
    )
    weekly = weekly.sort_values(["ppg_id", "week_start"])
    weekly["trailing_med"] = (
        weekly.groupby("ppg_id")["units"]
        .rolling(4, min_periods=2)
        .median()
        .reset_index(0, drop=True)
    )
    holiday_rows = weekly[weekly["holiday"] != ""].copy()
    holiday_rows["lift"] = (
        holiday_rows["units"] - holiday_rows["trailing_med"]
    ) / holiday_rows["trailing_med"].replace(0, np.nan)
    out: list[dict[str, Any]] = []
    for r in holiday_rows.itertuples(index=False):
        if pd.isna(r.lift):
            continue
        out.append(
            {
                "ppg_id": r.ppg_id,
                "week_start": r.week_start.isoformat() if hasattr(r.week_start, "isoformat") else str(r.week_start),
                "holiday": r.holiday,
                "units": int(r.units),
                "trailing_median": round(_safe_float(r.trailing_med), 2),
                "lift": round(_safe_float(r.lift), 3),
            }
        )
    out.sort(key=lambda r: abs(r["lift"]), reverse=True)
    return out


def cardinality_report(raw_panel: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Per categorical column: value counts + rare-flag at < 5% share."""
    out: list[dict[str, Any]] = []
    for c in columns:
        if c not in raw_panel.columns:
            continue
        s = raw_panel[c].astype(str).dropna()
        if s.empty:
            continue
        counts = s.value_counts()
        total = int(counts.sum())
        rows = []
        for v, n in counts.items():
            share = float(n) / total
            rows.append({"value": str(v), "n": int(n), "share": round(share, 4), "rare": share < 0.05})
        out.append({"column": c, "n_distinct": int(len(counts)), "values": rows})
    return out
