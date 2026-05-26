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
