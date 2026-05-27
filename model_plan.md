# Price & Promo Model Catalog

A comprehensive catalog of demand / elasticity / promo models we can build, the
problem each addresses, and the literature behind them. This is the menu the
model library (`core/models/library/`) and the model-selection router
(`core/models/router/`) draw from — it is intentionally exhaustive and does
**not** pre-filter for suitability.

~60 models across 12 families. ✓ = already implemented in `core/models/`.

## How the library uses this catalog

Each model becomes an independent plugin under `core/models/library/<family>/`
returning a uniform `ModelResult` (`core/models/result.py`). A router
(`core/models/router/`) chooses an ordered candidate set from this catalog based
on the declared **problem type** (own-price elasticity, cross-price
cannibalization, promo uplift, forecast, demand system) and an automatic
**data-diagnostics profile** (sample size, price variance, panel structure,
cross-price column availability, seasonality, target type), then escalates to
the next candidate when fit is poor.

---

## 1. Classical regression / econometric
| Model | Elasticity form | Notes |
|---|---|---|
| **Log-log OLS** ✓ | constant elasticity = β on log price | Workhorse; SCAN*PRO core |
| **Semi-log OLS** ✓ | ε = β·mean(price) | Sign-retry fallback |
| **Linear (level-level) OLS** | ε = β·(P̄/Q̄) | Simplest |
| **Log-linear / lin-log** | mixed | Asymmetric response |
| **SCAN*PRO** | multiplicative own+cross discount elasticities | Store-level promo standard; adds display/feature/seasonality dummies |
| **Polynomial / quadratic price** | ε = β₁ + 2β₂·logP | Curvature, satiation |
| **Box-Cox transformed regression** | data-driven transform | Generalizes log/linear |
| **Double-log with cross-price terms** | own + cross elasticities | Cannibalization |

## 2. Regularized regression
- **Ridge** — shrinks collinear coefficients (keeps all features)
- **Lasso** — shrinks + selects features
- **ElasticNet** — Ridge+Lasso blend; recommended for high-collinearity CPG data
- **Adaptive Lasso**, **Group Lasso** — grouped promo dummies

## 3. Panel / fixed-effects econometrics
- **Fixed-effects panel** (store/PPG/week dummies) — controls unobserved heterogeneity
- **Random-effects panel**
- **First-difference / within estimators**
- **Dynamic panel (Arellano–Bond GMM)** — lagged-dependent dynamics
- **Pooled OLS w/ clustered SE**

## 4. Instrumental variables / causal
- **2SLS / IV** — cost-shifter or Hausman instruments to fix price endogeneity
- **GMM**
- **Double/Debiased Machine Learning (DML)** — ML nuisance models + clean causal elasticity
- **Causal forests**, **Difference-in-Differences**, **Regression Discontinuity**

## 5. Discrete-choice / demand systems
- **Multinomial logit**
- **Nested logit**
- **Random-coefficients / mixed logit (BLP)** — realistic substitution, GMM-estimated
- **AIDS** (Almost Ideal Demand System) & **QUAIDS** (quadratic)
- **Rotterdam model**, **Translog demand system**

## 6. Bayesian
- **Hierarchical Bayesian regression** (partial pooling) — large uncertainty reduction on sparse items
- **Empirical Bayes shrinkage** ✓ (applied post-hoc today)
- **Bayesian structural time series (BSTS)**
- **Bayesian regression w/ informative priors**, **Bayesian model averaging**

## 7. Tree-based ensembles
- **LightGBM** ✓, **XGBoost**, **CatBoost**, **Random Forest**, **Gradient Boosting**, **Extra Trees**
- Elasticity recovered via price-bump simulation; monotone constraints on log_price

## 8. Other ML / nonparametric
- **Generalized Additive Models (GAMs)** — smooth nonlinear price response
- **Gaussian Process regression** — probabilistic, smooth
- **MARS** (multivariate adaptive regression splines)
- **Support Vector Regression**, **kernel/local-polynomial regression**, **kNN regression**

## 9. Time series (price/promo as exogenous regressors)
- **ARIMAX / SARIMAX**
- **Exponential smoothing (ETS / Holt-Winters)**, **TBATS**
- **State-space / Kalman filter / Dynamic Linear Models** — time-varying elasticity
- **VARX** (vector autoregression — cross-product dynamics)
- **Error-correction / cointegration models**
- **Prophet** (with price/promo regressors)

## 10. Deep learning
- **DeepAR** — probabilistic autoregressive RNN
- **LSTM / GRU**
- **Temporal Fusion Transformer (TFT)** — attention layers tuned for promo effects
- **N-BEATS / N-HiTS**, **Informer / Autoformer**
- **Graph Neural Networks** — cross-product substitution graph
- **WaveNet-style dilated CNN**

## 11. Promo-specific / baseline decomposition
- **Baseline + uplift decomposition** (multiplicative lift over base)
- **Marketing Mix Models (MMM)** — adstock/carryover + saturation (Hill/log) curves
- **Uplift / incrementality models** (two-model, causal trees)
- **Bass diffusion** & **Gompertz / logistic growth** — new-product launches

## 12. Robust / quantile
- **Huber M-estimator** ✓ (leverage-point refit)
- **RANSAC**, **Theil–Sen**, **Quantile regression**

---

## Framing notes
- **Cross-price / substitution** is native only to demand systems
  (logit/AIDS/BLP), VARX, and GNNs; the rest need explicit cross-price columns.
- **Probabilistic / uncertainty** out-of-the-box: Bayesian, GP, DeepAR, quantile.
- **Endogeneity-corrected** (cleanest causal elasticity): IV/2SLS, DML, BLP.

---

## Implementation status & how to add a model

Run `uv run automl models` to list every registered plugin with its family,
problem types, availability, and required packages (`--available-only` to hide
ones whose optional dependency isn't installed).

**Implemented (per-cell):** classical (`loglog_ols`, `semilog_ols`),
regularized (`ridge`, `lasso`, `elasticnet`), robust/quantile (`huber`,
`ransac`, `theil_sen`, `quantile`), trees (`lightgbm`, `random_forest`,
`extra_trees`, `xgboost`*, `catboost`*), ML/nonparam (`bayesian_ridge`,
`gaussian_process`, `svr`, `knn`, `gam`*), causal (`double_ml`), time-series
(`arimax`, `sarimax`, `state_space`, `ets`, `holt_winters`, `prophet`*,
`tbats`*). `*` = optional dependency, installed via a `models-*` extra.

**Not yet wired (need a multi-entity loop, not a single PPG slice):** panel
FE/RE, IV/2SLS, demand systems (logit/AIDS/BLP), VARX, GNN, hierarchical Bayes
(pymc), deep sequence models (DeepAR/LSTM/TFT — forecast path + torch).

### Adding a plugin
1. Create `core/models/library/<family>/<key>.py`. Subclass `BaseModelPlugin`,
   set `key`, `family`, `problem_types`, `capabilities`, `required_packages`,
   and implement `fit(frame, ctx) -> ModelResult`. Decorate with `@register`.
2. **Import heavy deps inside `fit`** (never at module top) so the library
   imports without them; `required_packages` drives the `is_available()` probe
   and the router drops unavailable models automatically.
3. A model module may import only `library.base`, `library.registry`,
   `models.result`, and shared `library._*` / `models.metrics` helpers — **never
   a sibling model module** (`tests/unit/test_library_no_cross_import.py`
   enforces this via AST).
4. Hyperparameters: return defaults from `default_hparams()` and read merged
   values via `self.resolve_hparams(ctx)` — no literals in `fit`. Per-family
   overrides live in `Settings.model_hparams`.
5. Register the module in its family `__init__`. Light (base-dep) families are
   imported eagerly in `core/models/library/__init__.py`; optional-dep families
   are imported defensively there.
6. Downstream compatibility: a linear-coefficient model (emits `coefficients`
   with `const`) should be added to `predictor.LINEAR_COEFF_MODELS`; a
   refit-scored model (trees/GP/SVR/kNN/GAM) to `predictor._REFIT_FACTORIES`
   (and thus `REFIT_MODELS`). Forecast-only models stay out of
   `PREDICTABLE_MODELS` and feed the FORECAST path's `forecasts.json`.
7. Add a unit test asserting sign recovery on synthetic and graceful skip when
   an optional dep is absent (`pytest.skip` guarded by `is_available()`).

## Sources
- [Mastering Price Elasticity Models for CPG (ElasticNet, RF, GBM)](https://medium.com/@quation755/mastering-price-elasticity-models-for-cpg-beyond-basics-eadfefbeb45d)
- [Modeling Price Elasticity of Demand — Strategic Brief (Revology/Kakas)](https://arminkakas.medium.com/modeling-price-elasticity-of-demand-a-strategic-brief-for-pricing-leaders-3fd7c109ad60)
- [Elasticity-Based Demand Forecasting and Price Optimization (arXiv)](https://arxiv.org/pdf/2106.08274)
- [Marketing Mix Modeling — Wikipedia (SCAN*PRO, adstock)](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
- [Marketing Models: Sales Promotion Models (SCAN*PRO)](https://medium.com/@adnan_ahmad/marketing-models-i-22cc82b3e32f)
- [Differentiated Products Demand Systems — Levin, Stanford (logit/nested/BLP/AIDS)](https://web.stanford.edu/~jdlevin/Econ%20257/Demand%20Estimation%20Slides%20B.pdf)
- [Foundations of Demand Estimation — Berry & Haile, Cowles](https://cowles.yale.edu/sites/default/files/2022-08/d2301_0.pdf)
- [Estimating Product-Level Price Elasticities Using Hierarchical Bayesian (TDS)](https://towardsdatascience.com/estimating-product-level-price-elasticities-using-hierarchical-bayesian/)
- [Double Machine Learning for Causal Inference and Price Elasticity](https://medium.com/@a.takeuchi121/double-machine-learning-for-causal-inference-and-price-elasticity-estimation-536c80d227ef)
- [Transformer-Based Probabilistic Time Series Forecasting with Explanatory Variables (MDPI)](https://www.mdpi.com/2227-7390/13/5/814)
- [Deep Demand Forecasting with Amazon SageMaker (DeepAR)](https://aws.amazon.com/blogs/machine-learning/deep-demand-forecasting-with-amazon-sagemaker/)
- [A Model to Improve the Estimation of Baseline Retail Sales](https://www.researchgate.net/publication/228118780_A_Model_to_Improve_the_Estimation_of_Baseline_Retail_Sales)
- [Bass Diffusion Model — Wikipedia](https://en.wikipedia.org/wiki/Bass_diffusion_model)
