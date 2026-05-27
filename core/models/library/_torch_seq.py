"""Shared torch sequence-forecaster helper for the deep family.

A small recurrent net (LSTM or GRU) forecasts ``log_units`` from a sliding
window of ``[log_units, log_price, *controls]``. The hold-out horizon is rolled
forward recursively using the KNOWN future exogenous values (price + controls)
and fed-back unit predictions. Forecast-only (no elasticity) — like the ETS
family — so it stays out of ``predictor.PREDICTABLE_MODELS``.

NOT a model module (registers nothing); model modules may import it. ``torch`` is
imported lazily inside ``fit_torch_seq`` so importing the library never requires
it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models.metrics import wape_units
from core.models.result import Capability, ForecastBlock, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"
TIME_COL = "week_start"


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def _index_labels(frame: pd.DataFrame, n: int) -> list[str]:
    if TIME_COL in frame.columns and len(frame):
        return [str(v) for v in frame[TIME_COL].astype(str).tolist()[:n]]
    return [str(i) for i in range(n)]


def fit_torch_seq(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    cell: str,
    test: pd.DataFrame | None = None,
    hparams: dict[str, Any] | None = None,
) -> ModelResult:
    import torch
    from torch import nn

    hp = {"lookback": 8, "hidden": 16, "epochs": 200, "lr": 0.01, "rng_seed": 0}
    hp.update(hparams or {})
    torch.manual_seed(int(hp["rng_seed"]))

    usable = _usable_controls(frame, controls)
    feat_cols = [TARGET, LOG_PRICE, *usable]
    train = frame[feat_cols].dropna().astype(float).to_numpy()
    lookback = int(hp["lookback"])
    if len(train) < lookback + 10:
        raise ValueError(f"too few rows for a sequence model ({len(train)})")

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    scaled = (train - mean) / std

    windows = np.stack([scaled[i - lookback : i] for i in range(lookback, len(scaled))])
    targets = scaled[lookback:, 0]  # next-step log_units (target is column 0)
    x = torch.tensor(windows, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU

    class _SeqNet(nn.Module):
        def __init__(self, n_features: int, hidden: int) -> None:
            super().__init__()
            self.rnn = rnn_cls(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, seq):  # noqa: ANN001
            out, _ = self.rnn(seq)
            return self.head(out[:, -1, :])

    model = _SeqNet(len(feat_cols), int(hp["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(hp["epochs"])):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    n_params = sum(p.numel() for p in model.parameters())
    diagnostics: dict[str, Any] = {"cell": cell, "n_params": int(n_params)}
    forecast: ForecastBlock | None = None
    if test is not None and len(test):
        tsub = test[feat_cols].dropna()
        if len(tsub):
            model.eval()
            window = scaled[-lookback:].copy()
            exog = ((tsub[[LOG_PRICE, *usable]].astype(float).to_numpy() - mean[1:]) / std[1:])
            preds_scaled: list[float] = []
            with torch.no_grad():
                for t in range(len(tsub)):
                    seq = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
                    p = float(model(seq).item())
                    preds_scaled.append(p)
                    nxt = np.concatenate([[p], exog[t]])
                    window = np.vstack([window[1:], nxt])
            mean_log = np.asarray(preds_scaled, dtype=float) * std[0] + mean[0]
            y_test = tsub[TARGET].astype(float).to_numpy()
            diagnostics["test_wape"] = wape_units(y_test, mean_log)
            diagnostics["n_test"] = int(len(tsub))
            forecast = ForecastBlock(
                horizon=len(mean_log),
                index=_index_labels(tsub, len(mean_log)),
                mean=[float(v) for v in np.exp(mean_log)],
            )

    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.FORECAST,
        own_elasticity=None,
        forecast=forecast,
        n_obs=int(len(train)),
        controls=usable,
        coefficients={},
        diagnostics=diagnostics,
        capabilities=Capability.FORECAST | Capability.NEEDS_TIME_INDEX | Capability.HEAVY_DEP,
    )
