"""Central runtime configuration.

Single source of truth for env-driven settings. Backed by ``pydantic-settings``
so every value is type-validated, documented in one place, and reachable from
both the FastAPI process and the CLI.

Access via :func:`get_settings` which caches a singleton for the process. Tests
that flip env vars must call ``get_settings.cache_clear()`` (the conftest
fixture does this automatically).

Per-agent model overrides are read from ``MODEL_<AGENT>`` env vars (e.g.
``MODEL_PPG_MAPPING=claude-sonnet-4-6``). The names match the agent's ``name``
attribute uppercased with underscores.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ModellingGrain(str, Enum):
    """Aggregation grain the modelling agent fits demand models at.

    Two-axis catalogue: the spatial axis is chain vs store, and the
    product axis is PPG vs category vs brand. Six combinations total.

    - ``ppg_week``: chain × PPG × week. Default; current behaviour.
    - ``store_ppg_week``: store × PPG × week. Hoch-style per-store fits.
    - ``category_week``: chain × category × week.
    - ``store_category_week``: store × category × week. Hoch (1995) grain.
    - ``brand_week``: chain × brand × week.
    - ``store_brand_week``: store × brand × week.

    At non-PPG grains, the engineered ``ppg_id`` column carries the
    brand or category label (not a PPG_AUTO_xx id) — downstream agents
    treat the string identifier opaquely and don't care which axis it
    represents.
    """

    PPG_WEEK = "ppg_week"
    STORE_PPG_WEEK = "store_ppg_week"
    CATEGORY_WEEK = "category_week"
    STORE_CATEGORY_WEEK = "store_category_week"
    BRAND_WEEK = "brand_week"
    STORE_BRAND_WEEK = "store_brand_week"


class ValidationThresholds(BaseModel):
    """Per-PPG verdict cutoffs used by ``core.validation.checks``."""

    sign_pass: float = 0.75
    sign_warn: float = 0.50
    wape_pass: float = 0.20
    wape_warn: float = 0.30
    cv_pass: float = 0.4
    cv_warn: float = 0.7
    elasticity_low: float = 0.3
    elasticity_high: float = 6.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None
    anthropic_auth_token: SecretStr | None = None
    llm_provider: Literal["", "dry_run", "api", "oauth", "cli"] = ""
    llm_dry_run: bool = False
    llm_trace: bool = True
    llm_cli_timeout_seconds: int = 180

    anthropic_model_opus: str = "claude-opus-4-7"
    anthropic_model_sonnet: str = "claude-sonnet-4-6"
    model_overrides: dict[str, str] = Field(default_factory=dict)

    run_dir: Path = Path("./runs")
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    api_auth_token: SecretStr | None = None
    max_upload_mb: int = 200

    api_proxy_target: str = "http://localhost:8000"

    baseline_dir: Path = Path("core/data/baselines")

    drift_slack_pct: float = 0.4

    validation: ValidationThresholds = ValidationThresholds()

    modelling_grain: ModellingGrain = ModellingGrain.PPG_WEEK

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    def model_post_init(self, __context: object) -> None:
        if not self.model_overrides:
            overrides: dict[str, str] = {}
            for key, value in os.environ.items():
                if not key.startswith("MODEL_") or not value:
                    continue
                if key in {"MODEL_OVERRIDES"}:
                    continue
                agent = key[len("MODEL_") :].lower()
                overrides[agent] = value
            self.model_overrides = overrides


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
